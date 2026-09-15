// Black-box integration tests for the cold-restart reconciliation path
// (WorkerPool.ReconcileIncompleteTrainings, orchestrator.GetModelStatus),
// against a fake in-memory S3 and fake gRPC workers.
//
// See test_suite_design.md, sezione D (D1-D5).
package orchestrator_test

import (
	"context"
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
		System:  testutil.NewTestSystemConfig(),
	}
}

func putMeta(t *testing.T, store *orchestrator.S3Store, modelID string, nEstimators int32) {
	t.Helper()
	meta := orchestrator.TrainRequestMetadata{
		DatasetURL:   "data/iris.csv",
		TaskType:     int32(pb.TaskType_CLASSIFICATION_TASK),
		TargetColumn: "target",
		NEstimators:  nEstimators,
	}
	require.NoError(t, store.PutJSON(context.Background(), "models/"+modelID+"/train_request.json", meta))
}

func putTree(t *testing.T, store *orchestrator.S3Store, modelID string, idx int32) {
	t.Helper()
	require.NoError(t, store.PutBytes(context.Background(), testutil.TreeKey(modelID, idx), []byte("tree")))
}

// ---------------------------------------------------------------------
// D1 — Recupero mirato dei soli alberi mancanti
// ---------------------------------------------------------------------

func TestD1_RecoverMissingTrees(t *testing.T) {
	fakeS3 := testutil.NewFakeS3(t)
	storageCfg := testutil.NewTestStorageConfig(fakeS3.Endpoint(), "bucket")
	ctx := context.Background()

	store, err := orchestrator.NewS3Store(ctx, &storageCfg)
	require.NoError(t, err)

	modelID := "model-d1"
	putMeta(t, store, modelID, 5)
	// Only trees 0 and 3 made it to S3 before the crash.
	putTree(t, store, modelID, 0)
	putTree(t, store, modelID, 3)

	w0 := testutil.NewFakeWorker(t)
	w0.TrainFunc = testutil.TrainFuncUploadingTrees(&storageCfg)
	w1 := testutil.NewFakeWorker(t)
	w1.TrainFunc = testutil.TrainFuncUploadingTrees(&storageCfg)

	pool := newPool(t, w0.Address, w1.Address)
	pool.ReconcileIncompleteTrainings(ctx, newReconcileConfig(storageCfg))

	// Only the 3 missing trees (1, 2, 4) were reissued - never 0 or 3.
	allCalls := append(w0.TrainCalls(), w1.TrainCalls()...)
	assert.Equal(t, []int32{1, 2, 4}, requestedTrees(allCalls))
	for _, c := range allCalls {
		assert.Equal(t, modelID, c.ModelId)
		assert.Equal(t, "data/iris.csv", c.DatasetUrl, "the training set comes from the persisted metadata")
	}

	// All 5 trees are now present and the model is ready.
	assert.Len(t, fakeS3.Keys("models/"+modelID+"/model_parts/"), 5)
	st, err := orchestrator.GetModelStatus(ctx, &storageCfg, modelID)
	require.NoError(t, err)
	assert.Equal(t, "ready", st.Status)
}

// ---------------------------------------------------------------------
// D2 — Recupero di un training interrotto prima di qualunque albero
// ---------------------------------------------------------------------

func TestD2_RecoverTrainingWithNoTreesYet(t *testing.T) {
	fakeS3 := testutil.NewFakeS3(t)
	storageCfg := testutil.NewTestStorageConfig(fakeS3.Endpoint(), "bucket")
	ctx := context.Background()

	store, err := orchestrator.NewS3Store(ctx, &storageCfg)
	require.NoError(t, err)

	modelID := "model-d2"
	// The master crashed right after persisting the request: no tree at all.
	// Whatever the number of workers healthy back then, the forest size is
	// fixed by the request.
	putMeta(t, store, modelID, 4)

	w0 := testutil.NewFakeWorker(t)
	w0.TrainFunc = testutil.TrainFuncUploadingTrees(&storageCfg)
	w1 := testutil.NewFakeWorker(t)
	w1.TrainFunc = testutil.TrainFuncUploadingTrees(&storageCfg)

	pool := newPool(t, w0.Address, w1.Address)
	pool.ReconcileIncompleteTrainings(ctx, newReconcileConfig(storageCfg))

	// Same model_id reused (not a new UUID), all 4 trees split over the 2
	// workers healthy now.
	require.Len(t, w0.TrainCalls(), 1)
	require.Len(t, w1.TrainCalls(), 1)
	assert.Equal(t, modelID, w0.TrainCalls()[0].ModelId)
	assert.Len(t, w0.TrainCalls()[0].TreeIndices, 2)
	assert.Len(t, w1.TrainCalls()[0].TreeIndices, 2)
	assert.Equal(t, indices(4), requestedTrees(append(w0.TrainCalls(), w1.TrainCalls()...)))

	var meta orchestrator.TrainRequestMetadata
	found, err := store.GetJSON(ctx, "models/"+modelID+"/train_request.json", &meta)
	require.NoError(t, err)
	require.True(t, found)
	assert.EqualValues(t, 4, meta.NEstimators, "the persisted request is never rewritten by recovery")

	assert.Len(t, fakeS3.Keys("models/"+modelID+"/model_parts/"), 4)
}

// ---------------------------------------------------------------------
// D3 — Più alberi mancanti, distribuiti su più worker sani; un modello
// marcato failed viene ritentato e il marker rimosso
// ---------------------------------------------------------------------

func TestD3_MultipleMissingTreesSpreadAcrossWorkers(t *testing.T) {
	fakeS3 := testutil.NewFakeS3(t)
	storageCfg := testutil.NewTestStorageConfig(fakeS3.Endpoint(), "bucket")
	ctx := context.Background()

	store, err := orchestrator.NewS3Store(ctx, &storageCfg)
	require.NoError(t, err)

	modelID := "model-d3"
	putMeta(t, store, modelID, 6)
	// Trees 0 and 1 present, 2..5 missing, and a previous attempt gave up.
	putTree(t, store, modelID, 0)
	putTree(t, store, modelID, 1)
	require.NoError(t, store.PutJSON(ctx, "models/"+modelID+"/training_failed.json",
		orchestrator.TrainingFailedMarker{Message: "gave up earlier"}))

	w0 := testutil.NewFakeWorker(t)
	w0.TrainFunc = testutil.TrainFuncUploadingTrees(&storageCfg)
	w1 := testutil.NewFakeWorker(t)
	w1.TrainFunc = testutil.TrainFuncUploadingTrees(&storageCfg)

	pool := newPool(t, w0.Address, w1.Address)
	pool.ReconcileIncompleteTrainings(ctx, newReconcileConfig(storageCfg))

	// The 4 missing trees are spread over the 2 workers, 2 each.
	require.Len(t, w0.TrainCalls(), 1)
	require.Len(t, w1.TrainCalls(), 1)
	assert.Len(t, w0.TrainCalls()[0].TreeIndices, 2)
	assert.Len(t, w1.TrainCalls()[0].TreeIndices, 2)
	assert.Equal(t, []int32{2, 3, 4, 5}, requestedTrees(append(w0.TrainCalls(), w1.TrainCalls()...)))

	assert.Len(t, fakeS3.Keys("models/"+modelID+"/model_parts/"), 6)

	// The forest is complete: the stale failure marker is gone.
	assert.False(t, fakeS3.Has("models/"+modelID+"/training_failed.json"))
	st, err := orchestrator.GetModelStatus(ctx, &storageCfg, modelID)
	require.NoError(t, err)
	assert.Equal(t, "ready", st.Status)
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
	putMeta(t, store, modelA, 3)
	putTree(t, store, modelA, 0)
	// Trees 1 and 2 missing for model A.

	modelB := "model-d4b"
	putMeta(t, store, modelB, 1)
	// Tree 0 missing for model B.

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
	// Each model exhausts its own retry budget (1 attempt + MaxRetriesPerTree)
	// and reconciliation moves on to the next one regardless.
	assert.Equal(t, 1+pool.MaxRetriesPerTree, modelACalls)
	assert.Equal(t, 1+pool.MaxRetriesPerTree, modelBCalls, "reconciliation moves on to model B despite model A's failures")

	// Nothing actually got recovered - the worker always fails - and both
	// models are reported as failed, with the reason.
	assert.Len(t, fakeS3.Keys("models/"+modelA+"/model_parts/"), 1, "still just the original tree 0")
	for _, id := range []string{modelA, modelB} {
		st, err := orchestrator.GetModelStatus(ctx, &storageCfg, id)
		require.NoError(t, err)
		assert.Equal(t, "failed", st.Status)
		assert.Contains(t, st.Message, "worker crashed during recovery")
	}
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
		assert.Equal(t, "not_found", got.Status)
	})

	t.Run("keys exist but train_request.json not written yet -> training (won't resolve on its own)", func(t *testing.T) {
		modelID := "model-d5-narrow-window"
		putTree(t, store, modelID, 0)

		got, err := orchestrator.GetModelStatus(ctx, &storageCfg, modelID)
		require.NoError(t, err)
		assert.Equal(t, "training", got.Status)
	})

	t.Run("some trees missing -> training", func(t *testing.T) {
		modelID := "model-d5-missing-trees"
		putMeta(t, store, modelID, 2)
		putTree(t, store, modelID, 0)

		got, err := orchestrator.GetModelStatus(ctx, &storageCfg, modelID)
		require.NoError(t, err)
		assert.Equal(t, "training", got.Status)
	})

	t.Run("some trees missing and a failure marker -> failed, with the message", func(t *testing.T) {
		modelID := "model-d5-failed"
		putMeta(t, store, modelID, 2)
		putTree(t, store, modelID, 0)
		require.NoError(t, store.PutJSON(ctx, "models/"+modelID+"/training_failed.json",
			orchestrator.TrainingFailedMarker{Message: "tree 1 failed 4 times"}))

		got, err := orchestrator.GetModelStatus(ctx, &storageCfg, modelID)
		require.NoError(t, err)
		assert.Equal(t, "failed", got.Status)
		assert.Equal(t, "tree 1 failed 4 times", got.Message)
	})

	t.Run("every tree present -> ready, even with a stale failure marker", func(t *testing.T) {
		modelID := "model-d5-ready"
		putMeta(t, store, modelID, 2)
		putTree(t, store, modelID, 0)
		putTree(t, store, modelID, 1)
		require.NoError(t, store.PutJSON(ctx, "models/"+modelID+"/training_failed.json",
			orchestrator.TrainingFailedMarker{Message: "stale"}))

		got, err := orchestrator.GetModelStatus(ctx, &storageCfg, modelID)
		require.NoError(t, err)
		assert.Equal(t, "ready", got.Status)
	})
}
