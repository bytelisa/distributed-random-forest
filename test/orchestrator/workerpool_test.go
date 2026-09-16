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
	"sort"
	"testing"

	pb "github.com/bytelisa/distributed-random-forest/api/proto/worker/v1"
	"github.com/bytelisa/distributed-random-forest/internal/orchestrator"
	"github.com/bytelisa/distributed-random-forest/test/testutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

func newPool(t *testing.T, addrs ...string) *orchestrator.WorkerPool {
	t.Helper()
	sys := testutil.NewTestSystemConfig()
	pool, err := orchestrator.NewWorkerPool(addrs, &sys)
	require.NoError(t, err)
	return pool
}

func newTrainJob(modelID string, nEstimators int32) orchestrator.TrainJob {
	return orchestrator.TrainJob{
		ModelID:      modelID,
		DatasetURL:   "data/iris.csv",
		TaskType:     pb.TaskType_CLASSIFICATION_TASK,
		TargetColumn: "target",
		NEstimators:  nEstimators,
	}
}

// requestedTrees flattens the tree indices of every Train call received.
func requestedTrees(calls []*pb.TrainRequest) []int32 {
	var all []int32
	for _, c := range calls {
		all = append(all, c.TreeIndices...)
	}
	sort.Slice(all, func(i, j int) bool { return all[i] < all[j] })
	return all
}

// predictedTrees flattens the tree indices of every Predict call received.
func predictedTrees(calls []*pb.PredictRequest) []int32 {
	var all []int32
	for _, c := range calls {
		all = append(all, c.TreeIndices...)
	}
	sort.Slice(all, func(i, j int) bool { return all[i] < all[j] })
	return all
}

func indices(n int32) []int32 {
	out := make([]int32, n)
	for i := range out {
		out[i] = int32(i)
	}
	return out
}

// ---------------------------------------------------------------------
// A3 — NewWorkerPool
// ---------------------------------------------------------------------

func TestA3_NewWorkerPool(t *testing.T) {
	sys := testutil.NewTestSystemConfig()

	t.Run("creates a client entry for every configured address, even unreachable ones", func(t *testing.T) {
		// grpc.NewClient dials lazily: it never fails at construction time
		// just because nothing is listening on an address yet - that's only
		// discovered later, at the first real RPC (health check).
		w1 := testutil.NewFakeWorker(t)
		unreachable := testutil.UnreachableAddress(t)

		pool, err := orchestrator.NewWorkerPool([]string{w1.Address, unreachable}, &sys)
		require.NoError(t, err)
		assert.Len(t, pool.Workers, 2)
	})

	t.Run("no addresses at all returns an explicit error", func(t *testing.T) {
		_, err := orchestrator.NewWorkerPool(nil, &sys)
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
	healthy.TrainFunc = testutil.TrainFuncUploadingTrees(&storageCfg)

	resp, err := pool.TrainDistributed(context.Background(), newTrainJob("model-c1", 5), &storageCfg)
	require.NoError(t, err)
	require.True(t, resp.Success, resp.Message)

	// The whole forest went to the only healthy worker: all 5 trees in one call.
	calls := healthy.TrainCalls()
	require.Len(t, calls, 1)
	assert.Equal(t, indices(5), calls[0].TreeIndices)
	assert.Equal(t, "data/iris.csv", calls[0].DatasetUrl, "every worker gets the full training set, not a partition")
}

// ---------------------------------------------------------------------
// C2 — Worker crashes during training (after health check)
// ---------------------------------------------------------------------

func TestC2_WorkerDiesDuringTrain(t *testing.T) {
	t.Run("no healthy peer: retries are exhausted, the model is marked failed and stays recoverable", func(t *testing.T) {
		crashing := testutil.NewFakeWorker(t)
		crashing.TrainFunc = func(_ context.Context, _ *pb.TrainRequest) (*pb.TrainResponse, error) {
			return nil, status.Error(codes.Unavailable, "worker process crashed")
		}

		pool := newPool(t, crashing.Address)

		fakeS3 := testutil.NewFakeS3(t)
		storageCfg := testutil.NewTestStorageConfig(fakeS3.Endpoint(), "bucket")

		resp, err := pool.TrainDistributed(context.Background(), newTrainJob("model-c2", 4), &storageCfg)

		// The call itself doesn't error out (TrainDistributed reports failure
		// via resp.Success, not via err).
		require.NoError(t, err)
		assert.False(t, resp.Success)
		assert.Contains(t, resp.Message, "giving up")

		// With a single worker, every retry lands on it again: 1 attempt + MaxRetriesPerTree.
		assert.Len(t, crashing.TrainCalls(), 1+pool.MaxRetriesPerTree)

		// The metadata is persisted before dispatching training, and the failure
		// is recorded, so GET /models/{id} reports it and a future
		// reconciliation pass can still pick the model up.
		assert.True(t, fakeS3.Has("models/model-c2/train_request.json"))
		st, err := orchestrator.GetModelStatus(context.Background(), &storageCfg, "model-c2")
		require.NoError(t, err)
		assert.Equal(t, "failed", st.Status)
		assert.Contains(t, st.Message, "giving up")
	})

	t.Run("a healthy peer is available: only the trees still missing are reassigned within the same call", func(t *testing.T) {
		fakeS3 := testutil.NewFakeS3(t)
		storageCfg := testutil.NewTestStorageConfig(fakeS3.Endpoint(), "bucket")
		upload := testutil.TrainFuncUploadingTrees(&storageCfg)

		ok := testutil.NewFakeWorker(t)
		ok.TrainFunc = upload

		// Uploads the first tree of its chunk, then dies.
		crashing := testutil.NewFakeWorker(t)
		crashing.TrainFunc = func(ctx context.Context, req *pb.TrainRequest) (*pb.TrainResponse, error) {
			first := &pb.TrainRequest{ModelId: req.ModelId, TreeIndices: req.TreeIndices[:1]}
			if _, err := upload(ctx, first); err != nil {
				return nil, err
			}
			return nil, status.Error(codes.Unavailable, "worker process crashed")
		}

		pool := newPool(t, ok.Address, crashing.Address)

		resp, err := pool.TrainDistributed(context.Background(), newTrainJob("model-c2b", 6), &storageCfg)
		require.NoError(t, err)
		require.True(t, resp.Success, resp.Message)

		// The crashing worker was asked once, for its 3 trees.
		crashingCalls := crashing.TrainCalls()
		require.Len(t, crashingCalls, 1)
		require.Len(t, crashingCalls[0].TreeIndices, 3)
		uploadedBeforeCrash := crashingCalls[0].TreeIndices[0]

		// The healthy worker got its own 3 trees, plus a retry carrying only
		// the 2 trees the crashing worker never uploaded.
		okCalls := ok.TrainCalls()
		require.Len(t, okCalls, 2)
		retry := okCalls[1]
		assert.Len(t, retry.TreeIndices, 2)
		assert.NotContains(t, retry.TreeIndices, uploadedBeforeCrash, "a tree already on S3 is not retrained")

		// Every tree of the forest is on S3 exactly once, and the model is ready.
		assert.Len(t, fakeS3.Keys("models/model-c2b/model_parts/"), 6)
		st, err := orchestrator.GetModelStatus(context.Background(), &storageCfg, "model-c2b")
		require.NoError(t, err)
		assert.Equal(t, "ready", st.Status)
		assert.False(t, fakeS3.Has("models/model-c2b/training_failed.json"))
	})
}

// ---------------------------------------------------------------------
// C3 — Predict with all healthy workers (happy path)
// ---------------------------------------------------------------------

func TestC3_PredictAllWorkersHealthy(t *testing.T) {
	fakeS3 := testutil.NewFakeS3(t)
	storageCfg := testutil.NewTestStorageConfig(fakeS3.Endpoint(), "bucket")
	store, err := orchestrator.NewS3Store(context.Background(), &storageCfg)
	require.NoError(t, err)
	require.NoError(t, testutil.SeedTrainedModel(context.Background(), store, "model-c3", 4))

	w0 := testutil.NewFakeWorker(t)
	w0.PredictFunc = testutil.PredictFuncReturning(testutil.ClassVote("1"))
	w1 := testutil.NewFakeWorker(t)
	w1.PredictFunc = testutil.PredictFuncReturning(&pb.TreePrediction{Classes: []string{"0", "1"}, Probabilities: []float64{0.6, 0.4}})

	pool := newPool(t, w0.Address, w1.Address)

	req := &pb.PredictRequest{ModelId: "model-c3", Features: []float32{1, 2, 3}}
	result, err := pool.PredictDistributed(context.Background(), req, "classification", &storageCfg)
	require.NoError(t, err)
	assert.Equal(t, "1", result.Value, "\"1\": 2*1.0 + 2*0.4 = 2.8 vs \"0\": 2*0.6 = 1.2")
	assert.Empty(t, result.Warning, "every tree answered, nothing to warn about")

	// The 4 trees are split between the 2 workers, 2 each, covering every
	// index exactly once. Which worker gets which chunk isn't guaranteed:
	// getHealthyWorkers health-checks concurrently and appends as each
	// goroutine finishes, so the order isn't tied to the configured one.
	require.Len(t, w0.PredictCalls(), 1)
	require.Len(t, w1.PredictCalls(), 1)
	assert.Len(t, w0.PredictCalls()[0].TreeIndices, 2)
	assert.Len(t, w1.PredictCalls()[0].TreeIndices, 2)
	assert.Equal(t, indices(4), predictedTrees(append(w0.PredictCalls(), w1.PredictCalls()...)))
}

// ---------------------------------------------------------------------
// C4 — Worker crashes during predict
// ---------------------------------------------------------------------

func TestC4_WorkerDiesDuringPredict(t *testing.T) {
	t.Run("one worker fails, its trees are reassigned to the survivor within the same request", func(t *testing.T) {
		fakeS3 := testutil.NewFakeS3(t)
		storageCfg := testutil.NewTestStorageConfig(fakeS3.Endpoint(), "bucket")
		store, err := orchestrator.NewS3Store(context.Background(), &storageCfg)
		require.NoError(t, err)
		require.NoError(t, testutil.SeedTrainedModel(context.Background(), store, "model-c4a", 4))

		ok := testutil.NewFakeWorker(t)
		ok.PredictFunc = testutil.PredictFuncReturning(testutil.ValueVote(42))
		crashing := testutil.NewFakeWorker(t)
		crashing.PredictFunc = func(_ context.Context, _ *pb.PredictRequest) (*pb.PredictResponse, error) {
			return nil, status.Error(codes.Unavailable, "worker crashed mid-predict")
		}

		pool := newPool(t, ok.Address, crashing.Address)

		req := &pb.PredictRequest{ModelId: "model-c4a", Features: []float32{1}}
		result, err := pool.PredictDistributed(context.Background(), req, "regression", &storageCfg)
		require.NoError(t, err)
		assert.Equal(t, "42.000000", result.Value)
		assert.Empty(t, result.Warning, "the retry recovered every tree, nothing missing")

		// The survivor answered for its own chunk and then for the crashed
		// worker's one: no tree of the forest was left out.
		require.Len(t, ok.PredictCalls(), 2)
		assert.Equal(t, indices(4), predictedTrees(ok.PredictCalls()))
	})

	t.Run("some trees exhaust their retries, the rest are still aggregated with a warning", func(t *testing.T) {
		fakeS3 := testutil.NewFakeS3(t)
		storageCfg := testutil.NewTestStorageConfig(fakeS3.Endpoint(), "bucket")
		store, err := orchestrator.NewS3Store(context.Background(), &storageCfg)
		require.NoError(t, err)
		require.NoError(t, testutil.SeedTrainedModel(context.Background(), store, "model-c4c", 6))

		// Every worker fails specifically on tree 4 (e.g. that one file is
		// unreachable), regardless of which one is asked - unlike the
		// subtest above, no retry can ever rescue it. With 6 trees over 2
		// workers, tree 4 falls in the chunk [3,4,5]: that whole chunk
		// fails together (retries operate at chunk granularity), while the
		// other chunk, [0,1,2], answers normally from any worker.
		const poisoned = int32(4)
		unreliable := func(_ context.Context, req *pb.PredictRequest) (*pb.PredictResponse, error) {
			for _, idx := range req.TreeIndices {
				if idx == poisoned {
					return nil, status.Error(codes.Unavailable, "tree file unreachable")
				}
			}
			resp := &pb.PredictResponse{}
			for range req.TreeIndices {
				resp.Predictions = append(resp.Predictions, testutil.ClassVote("cat"))
			}
			return resp, nil
		}

		w0 := testutil.NewFakeWorker(t)
		w0.PredictFunc = unreliable
		w1 := testutil.NewFakeWorker(t)
		w1.PredictFunc = unreliable

		pool := newPool(t, w0.Address, w1.Address)

		req := &pb.PredictRequest{ModelId: "model-c4c", Features: []float32{1}}
		result, err := pool.PredictDistributed(context.Background(), req, "classification", &storageCfg)
		require.NoError(t, err, "the chunk without the poisoned tree answered - not a total failure")
		assert.Equal(t, "cat", result.Value)

		require.NotEmpty(t, result.Warning)
		assert.Contains(t, result.Warning, "3 of 6 trees")
		assert.Contains(t, result.Warning, "[3 4 5]", "the whole chunk carrying tree 4 is named as missing")
	})

	t.Run("all workers fail, an explicit error is returned", func(t *testing.T) {
		fakeS3 := testutil.NewFakeS3(t)
		storageCfg := testutil.NewTestStorageConfig(fakeS3.Endpoint(), "bucket")
		store, err := orchestrator.NewS3Store(context.Background(), &storageCfg)
		require.NoError(t, err)
		require.NoError(t, testutil.SeedTrainedModel(context.Background(), store, "model-c4b", 2))

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
		_, err = pool.PredictDistributed(context.Background(), req, "regression", &storageCfg)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "giving up")
	})
}

// ---------------------------------------------------------------------
// C5 — Predict on non existent model_id
// ---------------------------------------------------------------------

func TestC5_PredictOnNonexistentModel(t *testing.T) {
	fakeS3 := testutil.NewFakeS3(t)
	storageCfg := testutil.NewTestStorageConfig(fakeS3.Endpoint(), "bucket")

	worker := testutil.NewFakeWorker(t)
	pool := newPool(t, worker.Address)

	req := &pb.PredictRequest{ModelId: "does-not-exist", Features: []float32{1, 2}}
	_, err := pool.PredictDistributed(context.Background(), req, "classification", &storageCfg)

	// The master knows from S3 that no training was ever requested for this
	// id: nothing is sent to the workers.
	require.ErrorIs(t, err, orchestrator.ErrModelNotFound)
	assert.Empty(t, worker.PredictCalls())
}

// ---------------------------------------------------------------------
// C6 — Predict with num_worker > trees: the master assigns no tree to the
// workers in excess, which are simply not called.
// ---------------------------------------------------------------------

func TestC6_MoreWorkersThanTrees(t *testing.T) {
	fakeS3 := testutil.NewFakeS3(t)
	storageCfg := testutil.NewTestStorageConfig(fakeS3.Endpoint(), "bucket")
	store, err := orchestrator.NewS3Store(context.Background(), &storageCfg)
	require.NoError(t, err)
	require.NoError(t, testutil.SeedTrainedModel(context.Background(), store, "model-c6", 2))

	workers := []*testutil.FakeWorker{testutil.NewFakeWorker(t), testutil.NewFakeWorker(t), testutil.NewFakeWorker(t)}
	for _, w := range workers {
		w.PredictFunc = testutil.PredictFuncReturning(testutil.ClassVote("cat"))
	}

	pool := newPool(t, workers[0].Address, workers[1].Address, workers[2].Address)

	req := &pb.PredictRequest{ModelId: "model-c6", Features: []float32{1}}
	result, err := pool.PredictDistributed(context.Background(), req, "classification", &storageCfg)
	require.NoError(t, err)
	assert.Equal(t, "cat", result.Value)

	var called int
	var all []*pb.PredictRequest
	for _, w := range workers {
		if len(w.PredictCalls()) > 0 {
			called++
			all = append(all, w.PredictCalls()...)
		}
	}
	assert.Equal(t, 2, called, "with 2 trees and 3 workers, one worker has nothing to do")
	assert.Equal(t, indices(2), predictedTrees(all))
}

// ---------------------------------------------------------------------
// C7 — Fake crashed worker was actually just slow:
// the deterministic filename (tree_{tree_index}.joblib, not a random UUID)
// means a late re-upload overwrites the same S3 key instead of creating a
// duplicate file for the same tree.
// ---------------------------------------------------------------------

func TestC7_LateReuploadOverwritesSameKey(t *testing.T) {
	fakeS3 := testutil.NewFakeS3(t)
	storageCfg := testutil.NewTestStorageConfig(fakeS3.Endpoint(), "bucket")
	store, err := orchestrator.NewS3Store(context.Background(), &storageCfg)
	require.NoError(t, err)

	key := testutil.TreeKey("model-c7", 0)

	// A worker completes a tree and uploads it.
	require.NoError(t, store.PutBytes(context.Background(), key, []byte("first-attempt")))

	// A health check later marks it "dead" and its tree gets reassigned
	// elsewhere - but in reality the original worker was just slow, and it
	// now also finishes and uploads under the exact same deterministic key.
	require.NoError(t, store.PutBytes(context.Background(), key, []byte("second-attempt")))

	keys, err := store.ListKeys(context.Background(), "models/model-c7/model_parts/")
	require.NoError(t, err)
	assert.Len(t, keys, 1, "same tree index must overwrite, never duplicate, the existing file")

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
		_, err := pool.TrainDistributed(context.Background(), newTrainJob("model-c8", 3), &storageCfg)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "no healthy workers available")
	})

	t.Run("PredictDistributed", func(t *testing.T) {
		req := &pb.PredictRequest{ModelId: "model-c8", Features: []float32{1}}
		_, err := pool.PredictDistributed(context.Background(), req, "classification", &storageCfg)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "no healthy workers available")
	})
}

// ---------------------------------------------------------------------
// C9 — Different number of workers in train vs predict
// ---------------------------------------------------------------------

func TestC9_WorkerCountChangesBetweenTrainAndPredict(t *testing.T) {
	fakeS3 := testutil.NewFakeS3(t)
	storageCfg := testutil.NewTestStorageConfig(fakeS3.Endpoint(), "bucket")

	w0 := testutil.NewFakeWorker(t)
	w1 := testutil.NewFakeWorker(t)
	w2 := testutil.NewFakeWorker(t)
	for _, w := range []*testutil.FakeWorker{w0, w1, w2} {
		w.TrainFunc = testutil.TrainFuncUploadingTrees(&storageCfg)
	}

	pool := newPool(t, w0.Address, w1.Address, w2.Address)

	// Train with all 3 workers healthy: 6 trees, 2 each.
	resp, err := pool.TrainDistributed(context.Background(), newTrainJob("model-c9", 6), &storageCfg)
	require.NoError(t, err)
	require.True(t, resp.Success)
	for _, w := range []*testutil.FakeWorker{w0, w1, w2} {
		require.Len(t, w.TrainCalls(), 1)
		assert.Len(t, w.TrainCalls()[0].TreeIndices, 2)
	}
	assert.Equal(t, indices(6), requestedTrees(append(append(w0.TrainCalls(), w1.TrainCalls()...), w2.TrainCalls()...)))

	// w2 goes down before the predict request.
	w2.SetHealthy(false)

	predictReq := &pb.PredictRequest{ModelId: "model-c9", Features: []float32{1}}
	_, err = pool.PredictDistributed(context.Background(), predictReq, "classification", &storageCfg)
	require.NoError(t, err)

	// The same 6 trees are re-split over just the 2 survivors, 3 each - a
	// worker doesn't need to have trained a tree to serve it.
	assert.Empty(t, w2.PredictCalls())
	require.Len(t, w0.PredictCalls(), 1)
	require.Len(t, w1.PredictCalls(), 1)
	assert.Len(t, w0.PredictCalls()[0].TreeIndices, 3)
	assert.Len(t, w1.PredictCalls()[0].TreeIndices, 3)
	assert.Equal(t, indices(6), predictedTrees(append(w0.PredictCalls(), w1.PredictCalls()...)))
}
