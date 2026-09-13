package orchestrator

import (
	"context"
	"fmt"

	"github.com/bytelisa/distributed-random-forest/internal/config"
)

// GetModelStatus reports the status of a single model by inspecting S3:
//   - "not_found": nothing in models/{model_id}/ on S3 at all.
//   - "ready": complete training, ready for inference.
//   - "training": incomplete training (either incomplete partitioning or training itself).
func GetModelStatus(ctx context.Context, storageCfg *config.StorageConfig, modelID string) (string, error) {
	store, err := NewS3Store(ctx, storageCfg)
	if err != nil {
		return "", err
	}

	// Check if model id is unknown
	modelKeys, err := store.ListKeys(ctx, fmt.Sprintf("models/%s/", modelID))
	if err != nil {
		return "", err
	}
	if len(modelKeys) == 0 {
		return "not_found", nil
	}

	var meta TrainRequestMetadata
	found, err := store.GetJSON(ctx, fmt.Sprintf("models/%s/train_request.json", modelID), &meta)
	if err != nil {
		return "", err
	}
	if !found {
		return "training", nil
	}

	modelPartKeys, err := store.ListKeys(ctx, fmt.Sprintf("models/%s/model_parts/", modelID))
	if err != nil {
		return "", err
	}

	present := make(map[int32]bool)
	for _, key := range modelPartKeys {
		if idx, ok := parsePartitionIndex(key); ok {
			present[idx] = true
		}
	}

	if int32(len(present)) >= meta.TotalPartitions {
		return "ready", nil
	}
	return "training", nil
}
