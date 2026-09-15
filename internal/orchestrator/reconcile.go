package orchestrator

import (
	"context"
	"log"
	"strings"
	"time"

	pb "github.com/bytelisa/distributed-random-forest/api/proto/worker/v1"
	"github.com/bytelisa/distributed-random-forest/internal/config"
)

// incompleteModel describes one training that a scan of S3 found left
// unfinished: some tree indices of the forest have no file on S3.
type incompleteModel struct {
	ModelID         string
	DatasetURL      string
	TaskType        int32
	TargetColumn    string
	NEstimators     int32
	Hyperparameters map[string]string
	MissingIndices  []int32
}

// FAULT TOLERANCE MID TRAINING
// ReconcileIncompleteTrainings looks for trainings left incomplete by a
// master crash and finishes them, by training just the trees missing on S3.
// Nothing about which worker had which tree survives a crash, so every tree
// index of the forest is checked, not just those of a "failed" worker.
//
// Models marked as failed by a previous attempt are retried too: the marker
// is removed if this attempt completes the forest.
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
		p.recoverMissingTrees(ctx, cfg, store, m)
	}
}

// findIncompleteModels scans S3 (read-only, never writes or dispatches
// training itself) and reports, for every model_id with train_request.json
// on S3, which tree indices are still missing.
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
		found, err := store.GetJSON(ctx, trainRequestKey(modelID), &meta)
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

		present, err := store.listTreeIndices(ctx, modelID)
		if err != nil {
			log.Printf("[Reconciler] Failed to list trees for model %s, skipping: %v", modelID, err)
			continue
		}

		var missing []int32
		for i := int32(0); i < meta.NEstimators; i++ {
			if !present[i] {
				missing = append(missing, i)
			}
		}

		if len(missing) > 0 {
			incomplete = append(incomplete, incompleteModel{
				ModelID:         modelID,
				DatasetURL:      meta.DatasetURL,
				TaskType:        meta.TaskType,
				TargetColumn:    meta.TargetColumn,
				NEstimators:     meta.NEstimators,
				Hyperparameters: meta.Hyperparameters,
				MissingIndices:  missing,
			})
		}
	}

	return incomplete, nil
}

// recoverMissingTrees trains the missing trees of a model on the workers
// healthy now, without touching the ones already on S3.
func (p *WorkerPool) recoverMissingTrees(ctx context.Context, cfg *config.Config, store *S3Store, m incompleteModel) {
	log.Printf("[Reconciler] Model %s is missing trees %v out of %d - reassigning.",
		m.ModelID, m.MissingIndices, m.NEstimators)

	trainCtx, cancel := context.WithTimeout(ctx, time.Duration(cfg.System.TimeoutTraining)*time.Second)
	defer cancel()

	activeWorkers, err := p.getHealthyWorkers(trainCtx)
	if err != nil {
		log.Printf("[Reconciler] No healthy workers available to recover model %s (will retry on next master startup): %v",
			m.ModelID, err)
		return
	}

	job := TrainJob{
		ModelID:         m.ModelID,
		DatasetURL:      m.DatasetURL,
		TaskType:        pb.TaskType(m.TaskType),
		TargetColumn:    m.TargetColumn,
		NEstimators:     m.NEstimators,
		Hyperparameters: m.Hyperparameters,
	}

	if err := p.runTraining(trainCtx, store, job, m.MissingIndices, activeWorkers); err != nil {
		log.Printf("[Reconciler] Failed to recover model %s (will retry on next master startup): %v", m.ModelID, err)
		return
	}

	log.Printf("[Reconciler] Model %s fully recovered.", m.ModelID)
}
