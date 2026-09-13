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

// FakePartitionDataset is an orchestrator.PartitionFunc that skips the real
// Python partitioner and uploads numWorkers placeholder dataset-partition
// files directly through S3Store. Fault-tolerance tests care about
// partition *counts* and *keys*, never the CSV content - that's
// pandas/scikit-learn territory, out of scope here.
func FakePartitionDataset(ctx context.Context, storageCfg *config.StorageConfig, req *pb.TrainRequest, numWorkers int) error {
	store, err := orchestrator.NewS3Store(ctx, storageCfg)
	if err != nil {
		return err
	}
	for i := 0; i < numWorkers; i++ {
		key := fmt.Sprintf("models/%s/dataset_partitions/part_%d.csv", req.ModelId, i)
		if err := store.PutBytes(ctx, key, []byte("fake,partition,data\n")); err != nil {
			return err
		}
	}
	return nil
}

// FailingPartitionDataset always fails, simulating a partitioner crash
// (e.g. bad dataset URL, source S3 unreachable).
func FailingPartitionDataset(_ context.Context, _ *config.StorageConfig, _ *pb.TrainRequest, _ int) error {
	return fmt.Errorf("fake partitioner failure")
}

// TrainFuncUploadingPart returns a FakeWorker.TrainFunc that mimics the
// real worker's final step on a successful Train (worker_service.py):
// uploading models/{model_id}/model_parts/forest_part_{worker_index}.joblib
// under a deterministic, index-based key. Needed by reconciliation tests
// that check the resulting S3 state, not just which RPCs were sent.
func TrainFuncUploadingPart(storageCfg *config.StorageConfig) func(context.Context, *pb.TrainRequest) (*pb.TrainResponse, error) {
	return func(ctx context.Context, req *pb.TrainRequest) (*pb.TrainResponse, error) {
		store, err := orchestrator.NewS3Store(ctx, storageCfg)
		if err != nil {
			return nil, err
		}
		key := fmt.Sprintf("models/%s/model_parts/forest_part_%d.joblib", req.ModelId, req.WorkerIndex)
		if err := store.PutBytes(ctx, key, []byte("fake-forest")); err != nil {
			return nil, err
		}
		return &pb.TrainResponse{Success: true, Message: "ok"}, nil
	}
}
