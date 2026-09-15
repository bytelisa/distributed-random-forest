package testutil

import (
	"context"
	"fmt"

	pb "github.com/bytelisa/distributed-random-forest/api/proto/worker/v1"
	"github.com/bytelisa/distributed-random-forest/internal/config"
	"github.com/bytelisa/distributed-random-forest/internal/orchestrator"
)

// NewTestStorageConfig points at a FakeS3 endpoint with dummy static
// credentials, so the AWS SDK's static-credentials path is used instead of
// its default chain (env vars / shared config / IMDS), which would be slow
// or fail outright in a test sandbox with no real AWS environment.
func NewTestStorageConfig(fakeS3Endpoint, bucket string) config.StorageConfig {
	return config.StorageConfig{
		Type:      "s3",
		Endpoint:  fakeS3Endpoint,
		AccessKey: "test",
		SecretKey: "test",
		Bucket:    bucket,
	}
}

// NewTestSystemConfig returns the system settings used by the tests: short
// timeouts, a few retries per tree and no real backoff wait.
func NewTestSystemConfig() config.SystemConfig {
	return config.SystemConfig{
		TimeoutTraining:     5,
		TimeoutPrediction:   5,
		TimeoutHealthCheck:  2,
		DefaultNEstimators:  10,
		MaxRetriesPerTree:   3,
		RetryBackoffSeconds: 0,
	}
}

// TreeKey is the S3 key the worker uploads one trained tree to.
func TreeKey(modelID string, treeIndex int32) string {
	return fmt.Sprintf("models/%s/model_parts/tree_%d.joblib", modelID, treeIndex)
}

// TrainFuncUploadingTrees returns a FakeWorker.TrainFunc that mimics the
// real worker's behavior on a successful Train (worker_service.py):
// uploading models/{model_id}/model_parts/tree_{index}.joblib for every
// requested tree index. Needed by tests that check the resulting S3 state,
// not just which RPCs were sent.
func TrainFuncUploadingTrees(storageCfg *config.StorageConfig) func(context.Context, *pb.TrainRequest) (*pb.TrainResponse, error) {
	return func(ctx context.Context, req *pb.TrainRequest) (*pb.TrainResponse, error) {
		store, err := orchestrator.NewS3Store(ctx, storageCfg)
		if err != nil {
			return nil, err
		}
		for _, idx := range req.TreeIndices {
			if err := store.PutBytes(ctx, TreeKey(req.ModelId, idx), []byte("fake-tree")); err != nil {
				return nil, err
			}
		}
		return &pb.TrainResponse{Success: true, Message: "ok"}, nil
	}
}

// ClassVote builds the TreePrediction of a classification tree that is
// certain about one class.
func ClassVote(class string) *pb.TreePrediction {
	return &pb.TreePrediction{Classes: []string{class}, Probabilities: []float64{1}}
}

// ValueVote builds the TreePrediction of a regression tree.
func ValueVote(value float64) *pb.TreePrediction {
	return &pb.TreePrediction{Value: value}
}

// PredictFuncReturning returns a FakeWorker.PredictFunc that answers every
// requested tree index with a copy of the given prediction.
func PredictFuncReturning(pred *pb.TreePrediction) func(context.Context, *pb.PredictRequest) (*pb.PredictResponse, error) {
	return func(_ context.Context, req *pb.PredictRequest) (*pb.PredictResponse, error) {
		resp := &pb.PredictResponse{}
		for range req.TreeIndices {
			resp.Predictions = append(resp.Predictions, &pb.TreePrediction{
				Classes:       pred.Classes,
				Probabilities: pred.Probabilities,
				Value:         pred.Value,
			})
		}
		return resp, nil
	}
}

// SeedTrainedModel writes to the fake S3 the state a completed training
// leaves behind: the metadata and one tree file per index.
func SeedTrainedModel(ctx context.Context, store *orchestrator.S3Store, modelID string, nEstimators int32) error {
	meta := orchestrator.TrainRequestMetadata{
		DatasetURL:   "data/iris.csv",
		TaskType:     int32(pb.TaskType_CLASSIFICATION_TASK),
		TargetColumn: "target",
		NEstimators:  nEstimators,
	}
	if err := store.PutJSON(ctx, "models/"+modelID+"/train_request.json", meta); err != nil {
		return err
	}
	for i := int32(0); i < nEstimators; i++ {
		if err := store.PutBytes(ctx, TreeKey(modelID, i), []byte("fake-tree")); err != nil {
			return err
		}
	}
	return nil
}
