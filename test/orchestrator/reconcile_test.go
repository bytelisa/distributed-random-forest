// Black-box integration tests for the cold-restart reconciliation path
// (WorkerPool.ReconcileIncompleteTrainings, orchestrator.GetModelStatus),
// against a fake in-memory S3 and fake gRPC workers.
//
// See test_suite_design.md, sezione D (D1-D5).
package orchestrator_test

import (
	"context"
	"fmt"
	"testing"

	pb "github.com/bytelisa/distributed-random-forest/api/proto/worker/v1"
	"github.com/bytelisa/distributed-random-forest/internal/config"
	"github.com/bytelisa/distributed-random-forest/internal/orchestrator"
	"github.com/bytelisa/distributed-random-forest/test/testutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

func newReconcileConfig(storageCfg config.StorageConfig) *config.Config {
	return &config.Config{
		Storage: storageCfg,
		System:  config.SystemConfig{TimeoutTraining: 5, TimeoutPrediction: 5, TimeoutHealthCheck: healthTimeoutSeconds},
	}
}

// ---------------------------------------------------------------------
// D1 — Recupero mirato delle sole partizioni mancanti (training_incomplete)
// ---------------------------------------------------------------------

func TestD1_RecoverMissingModelParts(t *testing.T) {
	fakeS3 := testutil.NewFakeS3(t)
	storageCfg := testutil.NewTestStorageConfig(fakeS3.Endpoint(), "bucket")
	ctx := context.Background()

	store, err := orchestrator.NewS3Store(ctx, &storageCfg)
	require.NoError(t, err)

	modelID := "model-d1"
	meta := orchestrator.TrainRequestMetadata{
		DatasetURL:      "data/iris.csv",
		TaskType:        int32(pb.TaskType_CLASSIFICATION_TASK),
		TargetColumn:    "target",
		NEstimators:     10,
		TotalPartitions: 3,
	}
	require.NoError(t, store.PutJSON(ctx, "models/"+modelID+"/train_request.json", meta))

	// Partitioning is complete: all 3 dataset partitions are present.
	for i := 0; i < 3; i++ {
		key := fmt.Sprintf("models/%s/dataset_partitions/part_%d.csv", modelID, i)
		require.NoError(t, store.PutBytes(ctx, key, []byte("data")))
	}
	// Only partition 0 was actually trained before the crash.
	require.NoError(t, store.PutBytes(ctx, fmt.Sprintf("models/%s/model_parts/forest_part_0.joblib", modelID), []byte("f0")))

	w0 := testutil.NewFakeWorker(t)
	w0.TrainFunc = testutil.TrainFuncUploadingPart(&storageCfg)
	w1 := testutil.NewFakeWorker(t)
	w1.TrainFunc = testutil.TrainFuncUploadingPart(&storageCfg)

	pool := newPool(t, w0.Address, w1.Address)
	pool.ReconcileIncompleteTrainings(ctx, newReconcileConfig(storageCfg))

	// Only the 2 missing partitions (1 and 2) were reissued - never 0.
	allCalls := append(w0.TrainCalls(), w1.TrainCalls()...)
	require.Len(t, allCalls, 2)

	var indices []int32
	for _, c := range allCalls {
		indices = append(indices, c.WorkerIndex)
		assert.Equal(t, int32(3), c.TotalWorkers, "total comes from the persisted metadata, not the current worker count")
		assert.Equal(t, modelID, c.ModelId)
	}
	assert.ElementsMatch(t, []int32{1, 2}, indices)

	// All 3 model parts are now present.
	keys, err := store.ListKeys(ctx, fmt.Sprintf("models/%s/model_parts/", modelID))
	require.NoError(t, err)
	assert.Len(t, keys, 3)

	// Dataset partitions were left untouched.
	dsKeys, err := store.ListKeys(ctx, fmt.Sprintf("models/%s/dataset_partitions/", modelID))
	require.NoError(t, err)
	assert.Len(t, dsKeys, 3)
}

// ---------------------------------------------------------------------
// D2 — Recupero di un training interrotto durante il partizionamento
// (partitioning_incomplete)
// ---------------------------------------------------------------------

func TestD2_RecoverIncompletePartitioning(t *testing.T) {
	fakeS3 := testutil.NewFakeS3(t)
	storageCfg := testutil.NewTestStorageConfig(fakeS3.Endpoint(), "bucket")
	ctx := context.Background()

	store, err := orchestrator.NewS3Store(ctx, &storageCfg)
	require.NoError(t, err)

	modelID := "model-d2"
	// The original attempt targeted 3 partitions (3 workers were healthy
	// back then), but the master crashed after uploading only 1 of them.
	meta := orchestrator.TrainRequestMetadata{
		DatasetURL:      "data/housing.csv",
		TaskType:        int32(pb.TaskType_REGRESSION_TASK),
		TargetColumn:    "target",
		NEstimators:     10,
		TotalPartitions: 3,
	}
	require.NoError(t, store.PutJSON(ctx, "models/"+modelID+"/train_request.json", meta))
	require.NoError(t, store.PutBytes(ctx, fmt.Sprintf("models/%s/dataset_partitions/part_0.csv", modelID), []byte("data")))
	// No model parts at all yet - never got that far.

	// Only 2 workers are healthy now, fewer than the stale 3 in metadata.
	w0 := testutil.NewFakeWorker(t)
	w0.TrainFunc = testutil.TrainFuncUploadingPart(&storageCfg)
	w1 := testutil.NewFakeWorker(t)
	w1.TrainFunc = testutil.TrainFuncUploadingPart(&storageCfg)

	pool := newPool(t, w0.Address, w1.Address)
	pool.ReconcileIncompleteTrainings(ctx, newReconcileConfig(storageCfg))

	// Repartitioned from scratch for the 2 CURRENTLY healthy workers, not
	// resumed from the stale 3.
	dsKeys, err := store.ListKeys(ctx, fmt.Sprintf("models/%s/dataset_partitions/", modelID))
	require.NoError(t, err)
	assert.Len(t, dsKeys, 2)

	var newMeta orchestrator.TrainRequestMetadata
	found, err := store.GetJSON(ctx, "models/"+modelID+"/train_request.json", &newMeta)
	require.NoError(t, err)
	require.True(t, found)
	assert.EqualValues(t, 2, newMeta.TotalPartitions, "metadata overwritten with the recomputed count, not the stale one")

	// Same model_id reused (not a new UUID) - both healthy workers trained
	// a full partition for it.
	require.Len(t, w0.TrainCalls(), 1)
	require.Len(t, w1.TrainCalls(), 1)
	assert.Equal(t, modelID, w0.TrainCalls()[0].ModelId)
	assert.Equal(t, int32(2), w0.TrainCalls()[0].TotalWorkers)

	modelKeys, err := store.ListKeys(ctx, fmt.Sprintf("models/%s/model_parts/", modelID))
	require.NoError(t, err)
	assert.Len(t, modelKeys, 2)
}

// ---------------------------------------------------------------------
// D3 — Più parti mancanti, distribuite su più worker sani
// ---------------------------------------------------------------------

func TestD3_MultipleMissingPartsSpreadAcrossWorkers(t *testing.T) {
	fakeS3 := testutil.NewFakeS3(t)
	storageCfg := testutil.NewTestStorageConfig(fakeS3.Endpoint(), "bucket")
	ctx := context.Background()

	store, err := orchestrator.NewS3Store(ctx, &storageCfg)
	require.NoError(t, err)

	modelID := "model-d3"
	meta := orchestrator.TrainRequestMetadata{TaskType: int32(pb.TaskType_CLASSIFICATION_TASK), TargetColumn: "t", NEstimators: 5, TotalPartitions: 4}
	require.NoError(t, store.PutJSON(ctx, "models/"+modelID+"/train_request.json", meta))
	for i := 0; i < 4; i++ {
		require.NoError(t, store.PutBytes(ctx, fmt.Sprintf("models/%s/dataset_partitions/part_%d.csv", modelID, i), []byte("d")))
	}
	// Parts 0 and 1 present, 2 and 3 missing.
	require.NoError(t, store.PutBytes(ctx, fmt.Sprintf("models/%s/model_parts/forest_part_0.joblib", modelID), []byte("f0")))
	require.NoError(t, store.PutBytes(ctx, fmt.Sprintf("models/%s/model_parts/forest_part_1.joblib", modelID), []byte("f1")))

	w0 := testutil.NewFakeWorker(t)
	w0.TrainFunc = testutil.TrainFuncUploadingPart(&storageCfg)
	w1 := testutil.NewFakeWorker(t)
	w1.TrainFunc = testutil.TrainFuncUploadingPart(&storageCfg)

	pool := newPool(t, w0.Address, w1.Address)
	pool.ReconcileIncompleteTrainings(ctx, newReconcileConfig(storageCfg))

	// The 2 missing indices (idx % len(activeWorkers)) land on 2 DIFFERENT
	// workers, never both on the same one.
	require.Len(t, w0.TrainCalls(), 1)
	require.Len(t, w1.TrainCalls(), 1)

	var indices []int32
	for _, c := range append(w0.TrainCalls(), w1.TrainCalls()...) {
		indices = append(indices, c.WorkerIndex)
		assert.Equal(t, int32(4), c.TotalWorkers)
	}
	assert.ElementsMatch(t, []int32{2, 3}, indices)

	keys, err := store.ListKeys(ctx, fmt.Sprintf("models/%s/model_parts/", modelID))
	require.NoError(t, err)
	assert.Len(t, keys, 4)
}

// ---------------------------------------------------------------------
// D4 — Un worker fallisce anche durante il tentativo di recupero
// ---------------------------------------------------------------------

func TestD4_WorkerFailsDuringRecovery(t *testing.T) {
	fakeS3 := testutil.NewFakeS3(t)
	storageCfg := testutil.NewTestStorageConfig(fakeS3.Endpoint(), "bucket")
	ctx := context.Background()

	store, err := orchestrator.NewS3Store(ctx, &storageCfg)
	require.NoError(t, err)

	modelA := "model-d4a"
	metaA := orchestrator.TrainRequestMetadata{TaskType: int32(pb.TaskType_CLASSIFICATION_TASK), TargetColumn: "t", NEstimators: 5, TotalPartitions: 3}
	require.NoError(t, store.PutJSON(ctx, "models/"+modelA+"/train_request.json", metaA))
	for i := 0; i < 3; i++ {
		require.NoError(t, store.PutBytes(ctx, fmt.Sprintf("models/%s/dataset_partitions/part_%d.csv", modelA, i), []byte("d")))
	}
	require.NoError(t, store.PutBytes(ctx, fmt.Sprintf("models/%s/model_parts/forest_part_0.joblib", modelA), []byte("f0")))
	// Indices 1 and 2 missing for model A.

	modelB := "model-d4b"
	metaB := orchestrator.TrainRequestMetadata{TaskType: int32(pb.TaskType_CLASSIFICATION_TASK), TargetColumn: "t", NEstimators: 5, TotalPartitions: 1}
	require.NoError(t, store.PutJSON(ctx, "models/"+modelB+"/train_request.json", metaB))
	require.NoError(t, store.PutBytes(ctx, fmt.Sprintf("models/%s/dataset_partitions/part_0.csv", modelB), []byte("d")))
	// Index 0 missing for model B.

	crashing := testutil.NewFakeWorker(t)
	crashing.TrainFunc = func(_ context.Context, _ *pb.TrainRequest) (*pb.TrainResponse, error) {
		return nil, status.Error(codes.Unavailable, "worker crashed during recovery")
	}

	pool := newPool(t, crashing.Address)

	assert.NotPanics(t, func() {
		pool.ReconcileIncompleteTrainings(ctx, newReconcileConfig(storageCfg))
	})

	var modelACalls, modelBCalls int
	for _, c := range crashing.TrainCalls() {
		switch c.ModelId {
		case modelA:
			modelACalls++
		case modelB:
			modelBCalls++
		}
	}
	assert.Equal(t, 2, modelACalls, "both missing partitions of model A are attempted even though each fails")
	assert.Equal(t, 1, modelBCalls, "reconciliation moves on to model B despite model A's failures")

	// Nothing actually got recovered - the worker always fails.
	keysA, err := store.ListKeys(ctx, fmt.Sprintf("models/%s/model_parts/", modelA))
	require.NoError(t, err)
	assert.Len(t, keysA, 1, "still just the original part 0 - the failed retries uploaded nothing")
}

// ---------------------------------------------------------------------
// D5 — GET /models/{id} coerente in ogni fase del recupero
// ---------------------------------------------------------------------

func TestD5_GetModelStatusAcrossRecoveryPhases(t *testing.T) {
	fakeS3 := testutil.NewFakeS3(t)
	storageCfg := testutil.NewTestStorageConfig(fakeS3.Endpoint(), "bucket")
	ctx := context.Background()

	store, err := orchestrator.NewS3Store(ctx, &storageCfg)
	require.NoError(t, err)

	t.Run("unknown model_id -> not_found", func(t *testing.T) {
		got, err := orchestrator.GetModelStatus(ctx, &storageCfg, "no-such-model")
		require.NoError(t, err)
		assert.Equal(t, "not_found", got)
	})

	t.Run("keys exist but train_request.json not written yet -> training (won't resolve on its own)", func(t *testing.T) {
		modelID := "model-d5-narrow-window"
		require.NoError(t, store.PutBytes(ctx, fmt.Sprintf("models/%s/dataset_partitions/part_0.csv", modelID), []byte("d")))

		got, err := orchestrator.GetModelStatus(ctx, &storageCfg, modelID)
		require.NoError(t, err)
		assert.Equal(t, "training", got)
	})

	t.Run("train_request.json present but dataset_partitions incomplete -> training", func(t *testing.T) {
		modelID := "model-d5-partitioning"
		meta := orchestrator.TrainRequestMetadata{TotalPartitions: 3}
		require.NoError(t, store.PutJSON(ctx, "models/"+modelID+"/train_request.json", meta))
		require.NoError(t, store.PutBytes(ctx, fmt.Sprintf("models/%s/dataset_partitions/part_0.csv", modelID), []byte("d")))

		got, err := orchestrator.GetModelStatus(ctx, &storageCfg, modelID)
		require.NoError(t, err)
		assert.Equal(t, "training", got)
	})

	t.Run("partitioning complete but some model parts missing -> training", func(t *testing.T) {
		modelID := "model-d5-missing-parts"
		meta := orchestrator.TrainRequestMetadata{TotalPartitions: 2}
		require.NoError(t, store.PutJSON(ctx, "models/"+modelID+"/train_request.json", meta))
		require.NoError(t, store.PutBytes(ctx, fmt.Sprintf("models/%s/model_parts/forest_part_0.joblib", modelID), []byte("f")))

		got, err := orchestrator.GetModelStatus(ctx, &storageCfg, modelID)
		require.NoError(t, err)
		assert.Equal(t, "training", got)
	})

	t.Run("every model part present -> ready", func(t *testing.T) {
		modelID := "model-d5-ready"
		meta := orchestrator.TrainRequestMetadata{TotalPartitions: 2}
		require.NoError(t, store.PutJSON(ctx, "models/"+modelID+"/train_request.json", meta))
		require.NoError(t, store.PutBytes(ctx, fmt.Sprintf("models/%s/model_parts/forest_part_0.joblib", modelID), []byte("f0")))
		require.NoError(t, store.PutBytes(ctx, fmt.Sprintf("models/%s/model_parts/forest_part_1.joblib", modelID), []byte("f1")))

		got, err := orchestrator.GetModelStatus(ctx, &storageCfg, modelID)
		require.NoError(t, err)
		assert.Equal(t, "ready", got)
	})
}
