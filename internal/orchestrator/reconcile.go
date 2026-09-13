package orchestrator

import (
	"context"
	"fmt"
	"log"
	"strings"
	"time"

	pb "github.com/bytelisa/distributed-random-forest/api/proto/worker/v1"
	"github.com/bytelisa/distributed-random-forest/internal/config"
)

// incompleteModel describes one training that a scan of S3 found left
// unfinished. TotalPartitions/MissingIndices are only populated when
// Status is "training_incomplete".
type incompleteModel struct {
	ModelID         string
	Status          string
	DatasetURL      string
	TaskType        int32
	TargetColumn    string
	NEstimators     int32
	TotalPartitions int32
	MissingIndices  []int32
}

// FAULT TOLERANCE MID TRAINING
// ReconcileIncompleteTrainings looks for trainings left incomplete by a
// master crash and finishes them.
//
// A model can be incomplete in one of two ways (see findIncompleteModels):
//   - "partitioning_incomplete": the crash happened before dataset
//     partitioning finished. Recovered by re-running the whole
//     TrainDistributed flow for that model_id, which also "refreshes" how
//     many partitions to use based on how many workers are healthy now.
//   - "training_incomplete": partitioning finished, but not every worker's
//     model part made it to S3. Recovered by reissuing training for just
//     the missing partitions.
//
// Models with no train_request.json at all are intentionally left out,
// it's a very specific case (basically immediate crash) where we have no info to resume the training.
// I decided that in this case it's up to the user to simply send another train req.
func (p *WorkerPool) ReconcileIncompleteTrainings(ctx context.Context, cfg *config.Config) {
	store, err := NewS3Store(ctx, &cfg.Storage)
	if err != nil {
		log.Printf("[Reconciler] Failed to create S3 client: %v", err)
		return
	}

	incomplete, err := findIncompleteModels(ctx, store)
	if err != nil {
		log.Printf("[Reconciler] Failed to check for incomplete trainings: %v", err)
		return
	}

	if len(incomplete) == 0 {
		log.Printf("[Reconciler] No incomplete trainings found.")
		return
	}

	for _, m := range incomplete {
		switch m.Status {
		case "partitioning_incomplete":
			p.recoverPartitioning(ctx, cfg, m)
		case "training_incomplete":
			p.recoverMissingParts(ctx, cfg, m)
		default:
			log.Printf("[Reconciler] Model %s reported with unknown status %q, skipping.", m.ModelID, m.Status)
		}
	}
}

// findIncompleteModels scans S3 (read-only, never writes or dispatches
// training itself) and reports, for every model_id with train_request.json
// on S3, whether partitioning or training is still incomplete.
func findIncompleteModels(ctx context.Context, store *S3Store) ([]incompleteModel, error) {
	prefixes, err := store.ListCommonPrefixes(ctx, "models/")
	if err != nil {
		return nil, err
	}

	var incomplete []incompleteModel
	for _, prefix := range prefixes {
		modelID := strings.TrimSuffix(strings.TrimPrefix(prefix, "models/"), "/")
		if modelID == "" {
			continue
		}

		var meta TrainRequestMetadata
		found, err := store.GetJSON(ctx, fmt.Sprintf("models/%s/train_request.json", modelID), &meta)
		if err != nil {
			log.Printf("[Reconciler] Failed to read metadata for model %s, skipping: %v", modelID, err)
			continue
		}
		if !found {
			// No metadata at all: too narrow a crash window (right after
			// computing how many workers are healthy, before that write
			// finishes) to recover from automatically.
			continue
		}

		// Stage 1: is partitioning itself done?
		partitionKeys, err := store.ListKeys(ctx, fmt.Sprintf("models/%s/dataset_partitions/", modelID))
		if err != nil {
			log.Printf("[Reconciler] Failed to list dataset partitions for model %s, skipping: %v", modelID, err)
			continue
		}

		if int32(len(partitionKeys)) < meta.TotalPartitions {
			incomplete = append(incomplete, incompleteModel{
				ModelID:      modelID,
				Status:       "partitioning_incomplete",
				DatasetURL:   meta.DatasetURL,
				TaskType:     meta.TaskType,
				TargetColumn: meta.TargetColumn,
				NEstimators:  meta.NEstimators,
			})
			continue
		}

		// Stage 2: partitioning is done, is every model part there?
		modelPartKeys, err := store.ListKeys(ctx, fmt.Sprintf("models/%s/model_parts/", modelID))
		if err != nil {
			log.Printf("[Reconciler] Failed to list model parts for model %s, skipping: %v", modelID, err)
			continue
		}

		present := make(map[int32]bool)
		for _, key := range modelPartKeys {
			if idx, ok := parsePartitionIndex(key); ok {
				present[idx] = true
			}
		}

		var missing []int32
		for i := int32(0); i < meta.TotalPartitions; i++ {
			if !present[i] {
				missing = append(missing, i)
			}
		}

		if len(missing) > 0 {
			incomplete = append(incomplete, incompleteModel{
				ModelID:         modelID,
				Status:          "training_incomplete",
				DatasetURL:      meta.DatasetURL,
				TaskType:        meta.TaskType,
				TargetColumn:    meta.TargetColumn,
				NEstimators:     meta.NEstimators,
				TotalPartitions: meta.TotalPartitions,
				MissingIndices:  missing,
			})
		}
	}

	return incomplete, nil
}

// recoverPartitioning redoes dataset partitioning (and the subsequent training dispatch) for a model
// whose partitioning never completed.
// It's the exact same flow as a brand new training request, just reusing the
// existing model_id instead of generating a new one, so that the user can use the received model_id and
// no new requests are needed.
// (resuming dataset partitioning through checkpointing was never an option :P )
func (p *WorkerPool) recoverPartitioning(ctx context.Context, cfg *config.Config, m incompleteModel) {
	log.Printf("[Reconciler] Model %s never finished partitioning - restarting it from scratch.", m.ModelID)

	req := &pb.TrainRequest{
		ModelId:      m.ModelID,
		DatasetUrl:   m.DatasetURL,
		TaskType:     pb.TaskType(m.TaskType),
		TargetColumn: m.TargetColumn,
		NEstimators:  m.NEstimators,
	}

	resp, err := p.TrainDistributed(ctx, req, &cfg.Storage)
	if err != nil || !resp.Success {
		log.Printf("[Reconciler] Failed to restart partitioning for model %s (will retry on next master startup): %v", m.ModelID, err)
		return
	}

	log.Printf("[Reconciler] Model %s fully recovered.", m.ModelID)
}

// recoverMissingParts reissues training for the specific partitions of a
// model that are missing a corresponding trained model part, without touching the ones already
// done.
func (p *WorkerPool) recoverMissingParts(ctx context.Context, cfg *config.Config, m incompleteModel) {
	log.Printf("[Reconciler] Model %s is missing partitions %v out of %d - reassigning.",
		m.ModelID, m.MissingIndices, m.TotalPartitions)

	datasetFolder := fmt.Sprintf("models/%s/dataset_partitions/", m.ModelID)
	trainTimeout := time.Duration(cfg.System.TimeoutTraining) * time.Second

	// Health-checked once for the whole model, not per missing index:
	// avoids that two different missing indices land on the same worker by chance
	activeWorkers, err := p.getHealthyWorkers(ctx)
	if err != nil {
		log.Printf("[Reconciler] No healthy workers available to recover model %s (will retry on next master startup): %v",
			m.ModelID, err)
		return
	}

	for _, idx := range m.MissingIndices {
		// Any healthy worker can take it: workers are stateless and
		// address dataset partitions/model parts by index.
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
