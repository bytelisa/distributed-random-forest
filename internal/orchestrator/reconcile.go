package orchestrator

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os/exec"
	"time"

	pb "github.com/bytelisa/distributed-random-forest/api/proto/worker/v1"
	"github.com/bytelisa/distributed-random-forest/internal/config"
)

// incompleteModel mirrors one entry of the JSON array printed by
// scripts/reconciler.py on stdout.
type incompleteModel struct {
	ModelID         string  `json:"model_id"`
	TaskType        int32   `json:"task_type"`
	TargetColumn    string  `json:"target_column"`
	NEstimators     int32   `json:"n_estimators"`
	TotalPartitions int32   `json:"total_partitions"`
	MissingIndices  []int32 `json:"missing_indices"`
}

// ReconcileIncompleteTrainings looks for trainings left incomplete by a
// master crash (dataset partitioning finished, but not every worker's
// model part uploaded on S3) and reissues training for the missing
// partitions on currently healthy workers.
//
// This is the master-side half of the cold standby recovery path: runs once,
// in the background, right after the new master starts.
// Models whose dataset partitioning never completed (no train_request.json on S3)
// are intentionally left alone here.
func (p *WorkerPool) ReconcileIncompleteTrainings(ctx context.Context, cfg *config.Config) {
	incomplete, err := listIncompleteModels(ctx, &cfg.Storage)
	if err != nil {
		log.Printf("[Reconciler] Failed to check for incomplete trainings: %v", err)
		return
	}

	if len(incomplete) == 0 {
		log.Printf("[Reconciler] No incomplete trainings found.")
		return
	}

	trainTimeout := time.Duration(cfg.System.TimeoutTraining) * time.Second

	for _, m := range incomplete {
		log.Printf("[Reconciler] Model %s is missing partitions %v out of %d - reassigning.",
			m.ModelID, m.MissingIndices, m.TotalPartitions)

		datasetFolder := fmt.Sprintf("models/%s/dataset_partitions/", m.ModelID)

		for _, idx := range m.MissingIndices {
			activeWorkers, err := p.getHealthyWorkers(ctx)
			if err != nil {
				log.Printf("[Reconciler] No healthy workers available to recover partition %d of model %s (will retry on next master startup): %v",
					idx, m.ModelID, err)
				continue
			}

			// Any healthy worker can take it: workers are stateless and
			// address dataset partitions/model parts by index, not by
			// identity. Spread across healthy workers if several parts
			// are missing at once.
			worker := activeWorkers[int(idx)%len(activeWorkers)]

			workerReq := &pb.TrainRequest{
				ModelId:      m.ModelID,
				DatasetUrl:   datasetFolder,
				TaskType:     pb.TaskType(m.TaskType),
				TargetColumn: m.TargetColumn,
				NEstimators:  m.NEstimators,
				WorkerIndex:  idx,
				TotalWorkers: m.TotalPartitions,
			}

			trainCtx, cancel := context.WithTimeout(ctx, trainTimeout)
			resp, err := worker.Client.Train(trainCtx, workerReq)
			cancel()

			if err != nil || !resp.Success {
				log.Printf("[Reconciler] Failed to recover partition %d of model %s on worker %s (will retry on next master startup): %v",
					idx, m.ModelID, worker.Address, err)
				continue
			}

			log.Printf("[Reconciler] Recovered partition %d of model %s on worker %s.", idx, m.ModelID, worker.Address)
		}
	}
}

// listIncompleteModels runs scripts/reconciler.py, which only reads S3 and reports, for every model whose
// dataset partitioning completed, which model part indices are still missing.
func listIncompleteModels(ctx context.Context, storageCfg *config.StorageConfig) ([]incompleteModel, error) {
	cmd := exec.CommandContext(ctx, "python", "scripts/reconciler.py",
		"--s3-endpoint", storageCfg.Endpoint,
		"--s3-access-key", storageCfg.AccessKey,
		"--s3-secret-key", storageCfg.SecretKey,
		"--s3-bucket", storageCfg.Bucket,
	)

	output, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("reconciler script failed: %w", err)
	}

	var incomplete []incompleteModel
	if err := json.Unmarshal(output, &incomplete); err != nil {
		return nil, fmt.Errorf("failed to parse reconciler output: %w", err)
	}

	return incomplete, nil
}
