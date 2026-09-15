// Package orchestrator_test holds black-box integration tests for
// WorkerPool: they only use its exported API (NewWorkerPool,
// TrainDistributed, PredictDistributed, ReconcileIncompleteTrainings,
// GetModelStatus), against fake gRPC workers (testutil.FakeWorker) and a
// fake in-memory S3 (testutil.FakeS3). No Python interpreter or real
// MinIO/S3 is needed to run these.
//
package orchestrator_test

import (
	"context"
	"testing"

	pb "github.com/bytelisa/distributed-random-forest/api/proto/worker/v1"
	"github.com/bytelisa/distributed-random-forest/internal/orchestrator"
	"github.com/bytelisa/distributed-random-forest/test/testutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

const healthTimeoutSeconds = 2

func newPool(t *testing.T, addrs ...string) *orchestrator.WorkerPool {
	t.Helper()
	pool, err := orchestrator.NewWorkerPool(addrs, healthTimeoutSeconds)
	require.NoError(t, err)
	pool.PartitionDataset = testutil.FakePartitionDataset
	return pool
}

func newTrainRequest(modelID string) *pb.TrainRequest {
	return &pb.TrainRequest{
		ModelId:      modelID,
		DatasetUrl:   "data/iris.csv",
		TaskType:     pb.TaskType_CLASSIFICATION_TASK,
		TargetColumn: "target",
		NEstimators:  10,
	}
}

// ---------------------------------------------------------------------
// A3 — NewWorkerPool
// ---------------------------------------------------------------------

func TestA3_NewWorkerPool(t *testing.T) {
	t.Run("creates a client entry for every configured address, even unreachable ones", func(t *testing.T) {
		// grpc.NewClient dials lazily: it never fails at construction time
		// just because nothing is listening on an address yet - that's only
		// discovered later, at the first real RPC (health check).
		w1 := testutil.NewFakeWorker(t)
		unreachable := testutil.UnreachableAddress(t)

		pool, err := orchestrator.NewWorkerPool([]string{w1.Address, unreachable}, healthTimeoutSeconds)
		require.NoError(t, err)
		assert.Len(t, pool.Workers, 2)
	})

	t.Run("no addresses at all returns an explicit error", func(t *testing.T) {
		_, err := orchestrator.NewWorkerPool(nil, healthTimeoutSeconds)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "no workers available")
	})
}

// ---------------------------------------------------------------------
// C1 — Worker unavailable to health check, before training
// ---------------------------------------------------------------------

func TestC1_UnreachableWorkerExcludedFromTrain(t *testing.T) {
	healthy := testutil.NewFakeWorker(t)
	unreachable := testutil.UnreachableAddress(t)

	pool := newPool(t, healthy.Address, unreachable)

	fakeS3 := testutil.NewFakeS3(t)
	storageCfg := testutil.NewTestStorageConfig(fakeS3.Endpoint(), "bucket")

	req := newTrainRequest("model-c1")
	resp, err := pool.TrainDistributed(context.Background(), req, &storageCfg)
	require.NoError(t, err)
	require.True(t, resp.Success, resp.Message)

	// Only the healthy worker should have been asked to train, indexed 0
	// out of a total of 1 (the unreachable one doesn't count).
	calls := healthy.TrainCalls()
	require.Len(t, calls, 1)
	assert.Equal(t, int32(0), calls[0].WorkerIndex)
	assert.Equal(t, int32(1), calls[0].TotalWorkers)

	// The dataset was partitioned into 1 part accordingly, not 2.
	assert.Len(t, fakeS3.Keys("models/model-c1/dataset_partitions/"), 1)
}

// ---------------------------------------------------------------------
// C2 — Worker crashes during training (after health check)
// ---------------------------------------------------------------------

func TestC2_WorkerDiesDuringTrain(t *testing.T) {
	t.Run("no healthy peer available: the whole call fails, but state is persisted for later reconciliation", func(t *testing.T) {
		crashing := testutil.NewFakeWorker(t)
		crashing.TrainFunc = func(_ context.Context, _ *pb.TrainRequest) (*pb.TrainResponse, error) {
			return nil, status.Error(codes.Unavailable, "worker process crashed")
		}

		pool := newPool(t, crashing.Address)

		fakeS3 := testutil.NewFakeS3(t)
		storageCfg := testutil.NewTestStorageConfig(fakeS3.Endpoint(), "bucket")

		req := newTrainRequest("model-c2")
		resp, err := pool.TrainDistributed(context.Background(), req, &storageCfg)

		// The call itself doesn't error out (TrainDistributed reports failure
		// via resp.Success, not via err). With only one worker configured,
		// there's no healthy peer to reassign the failed partition to (see
		// the subtest below for the case where there is one).
		require.NoError(t, err)
		assert.False(t, resp.Success)
		assert.Contains(t, resp.Message, "failed")

		// The metadata is persisted before dispatching training, so the model
		// stays visible to a future reconciliation pass as training_incomplete,
		// even though this synchronous call reported failure.
		assert.True(t, fakeS3.Has("models/model-c2/train_request.json"))
	})

	t.Run("a healthy peer is available: the failed partition is retried and reassigned within the same call", func(t *testing.T) {
		ok := testutil.NewFakeWorker(t)
		crashing := testutil.NewFakeWorker(t)
		crashing.TrainFunc = func(_ context.Context, _ *pb.TrainRequest) (*pb.TrainResponse, error) {
			return nil, status.Error(codes.Unavailable, "worker process crashed")
		}

		pool := newPool(t, ok.Address, crashing.Address)

		fakeS3 := testutil.NewFakeS3(t)
		storageCfg := testutil.NewTestStorageConfig(fakeS3.Endpoint(), "bucket")

		req := newTrainRequest("model-c2b")
		resp, err := pool.TrainDistributed(context.Background(), req, &storageCfg)
		require.NoError(t, err)

		// With a healthy peer available, the failed partition is retried on it
		// within the same call - no need to wait for a master restart.
		require.True(t, resp.Success, resp.Message)

		// The crashing worker was asked once, for its own partition, and failed.
		crashingCalls := crashing.TrainCalls()
		require.Len(t, crashingCalls, 1)

		// The healthy worker was asked twice: once for its own partition, once
		// (the retry) for the crashing worker's partition - same WorkerIndex as
		// the failed attempt, since partitions are addressed by index, not by
		// which physical worker handles them.
		okCalls := ok.TrainCalls()
		require.Len(t, okCalls, 2)
		gotIndices := []int32{okCalls[0].WorkerIndex, okCalls[1].WorkerIndex}
		assert.ElementsMatch(t, []int32{0, 1}, gotIndices, "the healthy worker should cover both partition indices between its own call and the retry")
	})
}

// ---------------------------------------------------------------------
// C3 — Predict with all healthy workers (happy path)
// ---------------------------------------------------------------------

func TestC3_PredictAllWorkersHealthy(t *testing.T) {
	w0 := testutil.NewFakeWorker(t)
	w0.PredictFunc = func(_ context.Context, _ *pb.PredictRequest) (*pb.PredictResponse, error) {
		return &pb.PredictResponse{Predictions: []string{"1", "1"}}, nil
	}
	w1 := testutil.NewFakeWorker(t)
	w1.PredictFunc = func(_ context.Context, _ *pb.PredictRequest) (*pb.PredictResponse, error) {
		return &pb.PredictResponse{Predictions: []string{"1", "0"}}, nil
	}

	pool := newPool(t, w0.Address, w1.Address)

	req := &pb.PredictRequest{ModelId: "model-c3", Features: []float32{1, 2, 3}}
	result, err := pool.PredictDistributed(context.Background(), req, "classification")
	require.NoError(t, err)
	assert.Equal(t, "1", result, "3 votes for \"1\" vs 1 for \"0\"")

	// Each worker gets a distinct index out of a total of 2. Which worker
	// gets which index isn't guaranteed: getHealthyWorkers health-checks
	// concurrently and appends as each goroutine finishes, so the order
	// isn't tied to the order addresses were configured in.
	require.Len(t, w0.PredictCalls(), 1)
	require.Len(t, w1.PredictCalls(), 1)
	assert.Equal(t, int32(2), w0.PredictCalls()[0].TotalWorkers)
	assert.Equal(t, int32(2), w1.PredictCalls()[0].TotalWorkers)
	assert.ElementsMatch(t, []int32{0, 1}, []int32{w0.PredictCalls()[0].WorkerIndex, w1.PredictCalls()[0].WorkerIndex})
}

// ---------------------------------------------------------------------
// C4 — Worker crashes during predict
// ---------------------------------------------------------------------

func TestC4_WorkerDiesDuringPredict(t *testing.T) {
	t.Run("one worker fails, the others' results are still aggregated", func(t *testing.T) {
		ok := testutil.NewFakeWorker(t)
		ok.PredictFunc = func(_ context.Context, _ *pb.PredictRequest) (*pb.PredictResponse, error) {
			return &pb.PredictResponse{Predictions: []string{"42.0"}}, nil
		}
		crashing := testutil.NewFakeWorker(t)
		crashing.PredictFunc = func(_ context.Context, _ *pb.PredictRequest) (*pb.PredictResponse, error) {
			return nil, status.Error(codes.Unavailable, "worker crashed mid-predict")
		}

		pool := newPool(t, ok.Address, crashing.Address)

		req := &pb.PredictRequest{ModelId: "model-c4a", Features: []float32{1}}
		result, err := pool.PredictDistributed(context.Background(), req, "regression")
		require.NoError(t, err)
		assert.Equal(t, "42.000000", result, "only the surviving worker's contribution is aggregated")
	})

	t.Run("all workers fail, an explicit error is returned", func(t *testing.T) {
		crashing1 := testutil.NewFakeWorker(t)
		crashing1.PredictFunc = func(_ context.Context, _ *pb.PredictRequest) (*pb.PredictResponse, error) {
			return nil, status.Error(codes.Unavailable, "crash 1")
		}
		crashing2 := testutil.NewFakeWorker(t)
		crashing2.PredictFunc = func(_ context.Context, _ *pb.PredictRequest) (*pb.PredictResponse, error) {
			return nil, status.Error(codes.Unavailable, "crash 2")
		}

		pool := newPool(t, crashing1.Address, crashing2.Address)

		req := &pb.PredictRequest{ModelId: "model-c4b", Features: []float32{1}}
		_, err := pool.PredictDistributed(context.Background(), req, "regression")
		require.Error(t, err)
		assert.Contains(t, err.Error(), "no workers returned valid results")
	})
}

// ---------------------------------------------------------------------
// C5 — Predict on non existent model_id
// ---------------------------------------------------------------------

func TestC5_PredictOnNonexistentModel(t *testing.T) {
	emptyWorker := testutil.NewFakeWorker(t)
	emptyWorker.PredictFunc = func(_ context.Context, _ *pb.PredictRequest) (*pb.PredictResponse, error) {
		// What worker_service.py does when it finds no model parts for the
		// given model_id: returns PredictResponse{} (predictions left as
		// its zero value, an empty list) - a valid, successful response
		// meaning "this worker contributes nothing", not an error.
		return &pb.PredictResponse{}, nil
	}

	pool := newPool(t, emptyWorker.Address)

	req := &pb.PredictRequest{ModelId: "does-not-exist", Features: []float32{1, 2}}
	_, err := pool.PredictDistributed(context.Background(), req, "classification")

	// The worker's response is successful but empty, so it contributes
	// nothing to the aggregation - since it was the only worker, the
	// aggregation step reports the generic "no results" error.
	require.Error(t, err)
	assert.Contains(t, err.Error(), "no workers returned valid results")
}

// ---------------------------------------------------------------------
// C6 — Predict with num_worker > trees: at least one
// worker ends up with zero files assigned by round-robin (more workers
// than trees), and correctly contributes nothing, while the others still
// succeed.
// ---------------------------------------------------------------------

func TestC6_MoreWorkersThanTrees(t *testing.T) {
	w0 := testutil.NewFakeWorker(t)
	w0.PredictFunc = func(_ context.Context, _ *pb.PredictRequest) (*pb.PredictResponse, error) {
		return &pb.PredictResponse{Predictions: []string{"cat"}}, nil
	}
	w1 := testutil.NewFakeWorker(t)
	w1.PredictFunc = func(_ context.Context, _ *pb.PredictRequest) (*pb.PredictResponse, error) {
		return &pb.PredictResponse{Predictions: []string{"cat"}}, nil
	}
	// Worker 2 got no files assigned by its own round-robin (more workers
	// than trees) - a valid, empty response, not an error.
	w2 := testutil.NewFakeWorker(t)
	w2.PredictFunc = func(_ context.Context, _ *pb.PredictRequest) (*pb.PredictResponse, error) {
		return &pb.PredictResponse{}, nil
	}

	pool := newPool(t, w0.Address, w1.Address, w2.Address)

	req := &pb.PredictRequest{ModelId: "model-c6", Features: []float32{1}}
	result, err := pool.PredictDistributed(context.Background(), req, "classification")
	require.NoError(t, err, "the other two workers still produced valid results")
	assert.Equal(t, "cat", result)
}

// ---------------------------------------------------------------------
// C7 — Fake crashed worker was actually just slow:
// the deterministic filename (forest_part_{worker_index}.joblib, not a
// random UUID) means a late re-upload overwrites the same S3 key instead
// of creating a duplicate part for the same partition.
// ---------------------------------------------------------------------

func TestC7_LateReuploadOverwritesSameKey(t *testing.T) {
	fakeS3 := testutil.NewFakeS3(t)
	storageCfg := testutil.NewTestStorageConfig(fakeS3.Endpoint(), "bucket")
	store, err := orchestrator.NewS3Store(context.Background(), &storageCfg)
	require.NoError(t, err)

	key := "models/model-c7/model_parts/forest_part_0.joblib"

	// A worker completes training and uploads its part.
	require.NoError(t, store.PutBytes(context.Background(), key, []byte("first-attempt")))

	// A health check later marks it "dead" and its partition gets
	// reassigned elsewhere - but in reality the original worker was just
	// slow, and it now also finishes and uploads under the exact same
	// deterministic key.
	require.NoError(t, store.PutBytes(context.Background(), key, []byte("second-attempt")))

	keys, err := store.ListKeys(context.Background(), "models/model-c7/model_parts/")
	require.NoError(t, err)
	assert.Len(t, keys, 1, "same worker_index must overwrite, never duplicate, the existing part")

	body, found, err := store.GetBytes(context.Background(), key)
	require.NoError(t, err)
	require.True(t, found)
	assert.Equal(t, "second-attempt", string(body))
}

// ---------------------------------------------------------------------
// C8 — No healthy worker available
// ---------------------------------------------------------------------

func TestC8_NoHealthyWorkers(t *testing.T) {
	dead := testutil.NewFakeWorker(t)
	dead.SetHealthy(false)

	pool := newPool(t, dead.Address)
	fakeS3 := testutil.NewFakeS3(t)
	storageCfg := testutil.NewTestStorageConfig(fakeS3.Endpoint(), "bucket")

	t.Run("TrainDistributed", func(t *testing.T) {
		_, err := pool.TrainDistributed(context.Background(), newTrainRequest("model-c8"), &storageCfg)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "no healthy workers available")
	})

	t.Run("PredictDistributed", func(t *testing.T) {
		req := &pb.PredictRequest{ModelId: "model-c8", Features: []float32{1}}
		_, err := pool.PredictDistributed(context.Background(), req, "classification")
		require.Error(t, err)
		assert.Contains(t, err.Error(), "no healthy workers available")
	})
}

// ---------------------------------------------------------------------
// C9 — Different number of workers in train vs predict
// ---------------------------------------------------------------------

func TestC9_WorkerCountChangesBetweenTrainAndPredict(t *testing.T) {
	w0 := testutil.NewFakeWorker(t)
	w1 := testutil.NewFakeWorker(t)
	w2 := testutil.NewFakeWorker(t)

	pool := newPool(t, w0.Address, w1.Address, w2.Address)
	fakeS3 := testutil.NewFakeS3(t)
	storageCfg := testutil.NewTestStorageConfig(fakeS3.Endpoint(), "bucket")

	// Train with all 3 workers healthy.
	resp, err := pool.TrainDistributed(context.Background(), newTrainRequest("model-c9"), &storageCfg)
	require.NoError(t, err)
	require.True(t, resp.Success)
	assert.Len(t, w0.TrainCalls(), 1)
	assert.Len(t, w1.TrainCalls(), 1)
	assert.Len(t, w2.TrainCalls(), 1)
	assert.Equal(t, int32(3), w0.TrainCalls()[0].TotalWorkers)

	// w2 goes down before the predict request.
	w2.SetHealthy(false)

	predictReq := &pb.PredictRequest{ModelId: "model-c9", Features: []float32{1}}
	_, err = pool.PredictDistributed(context.Background(), predictReq, "classification")
	require.NoError(t, err)

	// Indices/totals are recomputed fresh on just the 2 survivors - not the
	// 3 that trained the model.
	assert.Empty(t, w2.PredictCalls())
	require.Len(t, w0.PredictCalls(), 1)
	require.Len(t, w1.PredictCalls(), 1)
	assert.Equal(t, int32(2), w0.PredictCalls()[0].TotalWorkers)
	assert.Equal(t, int32(2), w1.PredictCalls()[0].TotalWorkers)
	assert.ElementsMatch(t, []int32{0, 1}, []int32{w0.PredictCalls()[0].WorkerIndex, w1.PredictCalls()[0].WorkerIndex})
}
