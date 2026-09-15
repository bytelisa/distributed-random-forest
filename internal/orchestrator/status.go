package orchestrator

import (
	"context"
	"fmt"

	"github.com/bytelisa/distributed-random-forest/internal/config"
)

// ModelStatus is what GET /models/{id} reports.
type ModelStatus struct {
	Status  string
	Message string
}

// GetModelStatus reports the status of a single model by inspecting S3:
//   - "not_found": nothing in models/{model_id}/ on S3 at all.
//   - "ready": every tree of the forest is on S3, ready for inference.
//   - "failed": the retry budget was exhausted, Message carries the last error.
//   - "training": anything else.
func GetModelStatus(ctx context.Context, storageCfg *config.StorageConfig, modelID string) (ModelStatus, error) {
	store, err := NewS3Store(ctx, storageCfg)
	if err != nil {
		return ModelStatus{}, err
	}

	// Check if model id is unknown
	modelKeys, err := store.ListKeys(ctx, fmt.Sprintf("models/%s/", modelID))
	if err != nil {
		return ModelStatus{}, err
	}
	if len(modelKeys) == 0 {
		return ModelStatus{Status: "not_found"}, nil
	}

	var meta TrainRequestMetadata
	found, err := store.GetJSON(ctx, trainRequestKey(modelID), &meta)
	if err != nil {
		return ModelStatus{}, err
	}
	if !found {
		return ModelStatus{Status: "training"}, nil
	}

	present, err := store.listTreeIndices(ctx, modelID)
	if err != nil {
		return ModelStatus{}, err
	}
	if int32(len(present)) >= meta.NEstimators {
		return ModelStatus{Status: "ready"}, nil
	}

	var failed TrainingFailedMarker
	found, err = store.GetJSON(ctx, trainingFailedKey(modelID), &failed)
	if err != nil {
		return ModelStatus{}, err
	}
	if found {
		return ModelStatus{Status: "failed", Message: failed.Message}, nil
	}

	return ModelStatus{Status: "training"}, nil
}
