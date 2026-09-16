package orchestrator

// Package orchestrator allows for an Orchestration Approach (centralized approach)
// Master uses Orchestrator to manage and coordinate a pool of Workers

import (
	"context"
	"errors"
	"fmt"
	"log"
	"sort"
	"strings"
	"sync"
	"time"

	pb "github.com/bytelisa/distributed-random-forest/api/proto/worker/v1"
	"github.com/bytelisa/distributed-random-forest/internal/config"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

// ErrModelNotFound is returned by PredictDistributed when no training
// metadata exists for the requested model_id.
var ErrModelNotFound = errors.New("model not found")

// WorkerClient wraps the gRPC client and the connection
type WorkerClient struct {
	Address string
	Client  pb.WorkerClient
	Conn    *grpc.ClientConn
}

// WorkerPool manages the list of connected workers
type WorkerPool struct {
	Workers            []*WorkerClient
	HealthCheckTimeout time.Duration
	MaxRetriesPerTree  int
	RetryBackoff       time.Duration
}

// NewWorkerPool initializes connections to all workers listed in the config
func NewWorkerPool(addresses []string, sys *config.SystemConfig) (*WorkerPool, error) {
	pool := &WorkerPool{
		Workers:            make([]*WorkerClient, 0, len(addresses)),
		HealthCheckTimeout: time.Duration(sys.TimeoutHealthCheck) * time.Second,
		MaxRetriesPerTree:  sys.MaxRetriesPerTree,
		RetryBackoff:       time.Duration(sys.RetryBackoffSeconds) * time.Second,
	}

	for _, addr := range addresses {
		// Create an insecure connection (for now)
		conn, err := grpc.NewClient(addr, grpc.WithTransportCredentials(insecure.NewCredentials()))
		if err != nil {
			// If one worker fails, we log it but don't stop the whole system (Fault Tolerance Start)
			log.Printf("[Orchestrator] Warning: Failed to connect to worker at %s: %v", addr, err)
			continue
		}

		client := pb.NewWorkerClient(conn)
		pool.Workers = append(pool.Workers, &WorkerClient{
			Address: addr,
			Client:  client,
			Conn:    conn,
		})
		log.Printf("[Orchestrator] Connected to worker at %s", addr)
	}

	if len(pool.Workers) == 0 {
		return nil, fmt.Errorf("no workers available")
	}

	return pool, nil
}

// getHealthyWorkers returns a list of workers that are currently alive
func (p *WorkerPool) getHealthyWorkers(ctx context.Context) ([]*WorkerClient, error) {

	var healthyWorkers []*WorkerClient
	var mu sync.Mutex
	var wg sync.WaitGroup

	// Ping all workers in parallel
	for _, w := range p.Workers {
		wg.Add(1)
		go func(worker *WorkerClient) {
			defer wg.Done()

			// Timeout
			shortCtx, cancel := context.WithTimeout(ctx, p.HealthCheckTimeout)
			defer cancel()

			resp, err := worker.Client.Health(shortCtx, &pb.HealthRequest{})
			if err == nil && resp.Healthy {
				mu.Lock()
				healthyWorkers = append(healthyWorkers, worker)
				mu.Unlock()
			} else {
				log.Printf("[Orchestrator] Worker at %s is UNREACHABLE/UNHEALTHY.", worker.Address)
			}
		}(w)
	}

	wg.Wait()

	if len(healthyWorkers) == 0 {
		return nil, fmt.Errorf("critical: no healthy workers available")
	}

	return healthyWorkers, nil
}

// waitForReplacement returns a healthy worker to hand a failed chunk of
// trees to, preferring one other than the worker that just failed. While no
// worker is healthy it waits RetryBackoff between health checks, until ctx expires.
func (p *WorkerPool) waitForReplacement(ctx context.Context, failed *WorkerClient) (*WorkerClient, error) {
	for {
		healthy, err := p.getHealthyWorkers(ctx)
		if err == nil {
			for _, w := range healthy {
				if w.Address != failed.Address {
					return w, nil
				}
			}
			return healthy[0], nil
		}

		log.Printf("[Orchestrator] No healthy worker available, retrying in %s.", p.RetryBackoff)
		select {
		case <-ctx.Done():
			return nil, fmt.Errorf("no healthy worker became available: %w", ctx.Err())
		case <-time.After(p.RetryBackoff):
		}
	}
}

// splitIndices splits indices into n contiguous chunks, as even as possible
// (the remainder goes to the first chunks). Chunks may be empty when there
// are more workers than indices.
func splitIndices(indices []int32, n int) [][]int32 {
	chunks := make([][]int32, n)
	base := len(indices) / n
	rest := len(indices) % n
	start := 0
	for i := 0; i < n; i++ {
		size := base
		if i < rest {
			size++
		}
		chunks[i] = indices[start : start+size]
		start += size
	}
	return chunks
}

func rangeIndices(n int32) []int32 {
	indices := make([]int32, n)
	for i := range indices {
		indices[i] = int32(i)
	}
	return indices
}

// TrainJob describes one forest to train. NEstimators is the total number
// of trees: the master decides which worker builds which of them.
type TrainJob struct {
	ModelID         string
	DatasetURL      string
	TaskType        pb.TaskType
	TargetColumn    string
	NEstimators     int32
	Hyperparameters map[string]string
}

// request builds the gRPC message asking a worker for a subset of the trees.
func (j TrainJob) request(indices []int32) *pb.TrainRequest {
	return &pb.TrainRequest{
		ModelId:         j.ModelID,
		DatasetUrl:      j.DatasetURL,
		TaskType:        j.TaskType,
		TargetColumn:    j.TargetColumn,
		Hyperparameters: j.Hyperparameters,
		TreeIndices:     indices,
	}
}

// TrainDistributed persists the job, splits the forest's trees among the
// healthy workers and waits for every tree to be on S3.
func (p *WorkerPool) TrainDistributed(ctx context.Context, job TrainJob, storageCfg *config.StorageConfig) (*pb.TrainResponse, error) {

	activeWorkers, err := p.getHealthyWorkers(ctx)
	if err != nil {
		return nil, fmt.Errorf("training failed: %w", err)
	}

	// 1. PERSIST TRAIN REQUEST METADATA
	// Immediately for fault tolerance
	store, err := NewS3Store(ctx, storageCfg)
	if err != nil {
		return nil, fmt.Errorf("failed to create S3 client: %w", err)
	}

	meta := TrainRequestMetadata{
		DatasetURL:      job.DatasetURL,
		TaskType:        int32(job.TaskType),
		TargetColumn:    job.TargetColumn,
		NEstimators:     job.NEstimators,
		Hyperparameters: job.Hyperparameters,
	}
	if err := store.PutJSON(ctx, trainRequestKey(job.ModelID), meta); err != nil {
		return nil, fmt.Errorf("failed to persist train request metadata: %w", err)
	}

	// 2. DISTRIBUTE THE TREES
	log.Printf("[Orchestrator] Distributing %d trees of model %s over %d workers.", job.NEstimators, job.ModelID, len(activeWorkers))

	if err := p.runTraining(ctx, store, job, rangeIndices(job.NEstimators), activeWorkers); err != nil {
		return &pb.TrainResponse{
			Success: false,
			Message: fmt.Sprintf("Distributed training failed mid-process. Error: %v", err),
		}, nil
	}

	return &pb.TrainResponse{
		Success: true,
		Message: fmt.Sprintf("Training completed successfully on %d workers.", len(activeWorkers)),
	}, nil
}

// runTraining trains the given trees and keeps the failure marker of the
// model in sync with the outcome: written when the retry budget is
// exhausted, removed once the trees are all on S3.
func (p *WorkerPool) runTraining(ctx context.Context, store *S3Store, job TrainJob, indices []int32, workers []*WorkerClient) error {
	if err := p.trainTrees(ctx, store, job, indices, workers); err != nil {
		if markErr := store.PutJSON(ctx, trainingFailedKey(job.ModelID), TrainingFailedMarker{Message: err.Error()}); markErr != nil {
			log.Printf("[Orchestrator] Failed to persist failure marker for model %s: %v", job.ModelID, markErr)
		}
		return err
	}
	if err := store.DeleteKey(ctx, trainingFailedKey(job.ModelID)); err != nil {
		log.Printf("[Orchestrator] Failed to clear failure marker for model %s: %v", job.ModelID, err)
	}
	return nil
}

// trainTrees hands one contiguous chunk of tree indices to each worker, in
// parallel, and returns once every chunk either completed or gave up.
func (p *WorkerPool) trainTrees(ctx context.Context, store *S3Store, job TrainJob, indices []int32, workers []*WorkerClient) error {
	chunks := splitIndices(indices, len(workers))

	var wg sync.WaitGroup
	errChan := make(chan error, len(workers))

	for i, worker := range workers {
		if len(chunks[i]) == 0 {
			continue
		}
		wg.Add(1)
		go func(w *WorkerClient, chunk []int32) {
			defer wg.Done()
			if err := p.trainChunkWithRetry(ctx, store, job, chunk, w); err != nil {
				errChan <- err
			}
		}(worker, chunks[i])
	}

	wg.Wait()
	close(errChan)

	var msgs []string
	for err := range errChan {
		msgs = append(msgs, err.Error())
	}
	if len(msgs) > 0 {
		return errors.New(strings.Join(msgs, "; "))
	}
	return nil
}

// trainChunkWithRetry asks worker to train chunk. When the call fails, only
// the trees that didn't make it to S3 are reassigned to another healthy
// worker, each tree at most MaxRetriesPerTree times.
func (p *WorkerPool) trainChunkWithRetry(ctx context.Context, store *S3Store, job TrainJob, chunk []int32, worker *WorkerClient) error {
	attempts := make(map[int32]int, len(chunk))
	pending := chunk

	for {
		for _, idx := range pending {
			attempts[idx]++
		}

		log.Printf("[Orchestrator] Sending %d trees of model %s to worker %s.", len(pending), job.ModelID, worker.Address)
		resp, err := worker.Client.Train(ctx, job.request(pending))
		if err == nil && resp.Success {
			return nil
		}
		failure := describeFailure(err, resp)
		log.Printf("[Orchestrator] Worker %s failed on model %s: %s", worker.Address, job.ModelID, failure)

		// Trees uploaded before the failure don't need to be retrained
		present, err := store.listTreeIndices(ctx, job.ModelID)
		if err != nil {
			return fmt.Errorf("model %s: failed to check trees on S3 after worker failure: %w", job.ModelID, err)
		}
		var missing []int32
		for _, idx := range pending {
			if !present[idx] {
				missing = append(missing, idx)
			}
		}
		if len(missing) == 0 {
			return nil
		}

		for _, idx := range missing {
			if attempts[idx] > p.MaxRetriesPerTree {
				return fmt.Errorf("tree %d of model %s failed %d times, giving up: %s", idx, job.ModelID, attempts[idx], failure)
			}
		}

		replacement, err := p.waitForReplacement(ctx, worker)
		if err != nil {
			return fmt.Errorf("model %s: trees %v could not be reassigned: %w", job.ModelID, missing, err)
		}
		log.Printf("[Orchestrator] Reassigning trees %v of model %s to worker %s.", missing, job.ModelID, replacement.Address)
		pending = missing
		worker = replacement
	}
}

func describeFailure(err error, resp *pb.TrainResponse) string {
	if err != nil {
		return err.Error()
	}
	if resp != nil {
		return resp.Message
	}
	return "unknown failure"
}

// THOUGHTS:
// Call Predict goroutines on the (correct!) workers
// Note: should make sure that we ask to predict to the workers who actually trained the model?
// Or maybe it's not relevant because the trained model is available on shared storage so any worker can use any part to contribute to the prediction?
// YES: final decision went on stateless workers, so the model "parts" (trained trees) are partitioned between the workers who simply download them.
// A worker can use trees he didn't train for inference purposes
// This also solves (partly) fault tolerance --> no lost state (no cached trained trees)

// PredictResult is the outcome of a distributed prediction: Value is the
// aggregated prediction, Warning is non-empty when it was computed on fewer
// trees than the forest actually has, because one or more chunks exhausted
// their retries.
type PredictResult struct {
	Value   string
	Warning string
}

// predictChunkOutcome is what one worker's chunk of trees resolved to: the
// predictions from any workers that answered, or the last error if none did.
type predictChunkOutcome struct {
	indices []int32
	preds   []*pb.TreePrediction
	err     error
}

// PredictDistributed splits the forest's trees among the healthy workers,
// collects one prediction per tree and aggregates them (Bagging - Aggregation Phase).
// A worker failure does not fail the whole request as long as at least one
// tree's prediction comes back: the aggregation proceeds on whatever
// arrived, and PredictResult.Warning reports what was skipped and why.
// taskType should be "classification" or "regression"
func (p *WorkerPool) PredictDistributed(ctx context.Context, req *pb.PredictRequest, taskType string, storageCfg *config.StorageConfig) (PredictResult, error) {

	// DYNAMIC HEALTH CHECK
	activeWorkers, err := p.getHealthyWorkers(ctx)
	if err != nil {
		return PredictResult{}, err
	}
	numWorkers := len(activeWorkers)
	log.Printf("[Orchestrator] Active workers for inference: %d (configured: %d)", numWorkers, len(p.Workers))

	// The forest size comes from the persisted request, the trees must all be there
	store, err := NewS3Store(ctx, storageCfg)
	if err != nil {
		return PredictResult{}, fmt.Errorf("failed to create S3 client: %w", err)
	}
	var meta TrainRequestMetadata
	found, err := store.GetJSON(ctx, trainRequestKey(req.ModelId), &meta)
	if err != nil {
		return PredictResult{}, fmt.Errorf("failed to read metadata of model %s: %w", req.ModelId, err)
	}
	if !found {
		return PredictResult{}, ErrModelNotFound
	}
	present, err := store.listTreeIndices(ctx, req.ModelId)
	if err != nil {
		return PredictResult{}, fmt.Errorf("failed to list trees of model %s: %w", req.ModelId, err)
	}
	if int32(len(present)) < meta.NEstimators {
		return PredictResult{}, fmt.Errorf("model %s is not ready: %d of %d trees missing", req.ModelId, meta.NEstimators-int32(len(present)), meta.NEstimators)
	}

	chunks := splitIndices(rangeIndices(meta.NEstimators), numWorkers)
	outcomes := make(chan predictChunkOutcome, numWorkers)
	var wg sync.WaitGroup

	log.Printf("[Orchestrator] Broadcasting prediction request to %d workers...", numWorkers)

	for i, worker := range activeWorkers {
		if len(chunks[i]) == 0 {
			continue
		}
		wg.Add(1)
		go func(w *WorkerClient, chunk []int32) {
			defer wg.Done()
			preds, err := p.predictChunkWithRetry(ctx, req, chunk, w)
			outcomes <- predictChunkOutcome{indices: chunk, preds: preds, err: err}
		}(worker, chunks[i])
	}

	wg.Wait()
	close(outcomes)

	// COLLECT PREDICTIONS
	// We merge all partial lists into one global list of votes/values.
	// A chunk that exhausted its retries does not abort the whole request:
	// its trees are simply missing from the aggregation, reported via Warning.
	// Note: Only aggregate once (no local aggregation on worker) in order to introduce less error.
	var globalPredictions []*pb.TreePrediction
	var missingIndices []int32
	var lastErr error
	for o := range outcomes {
		if o.err != nil {
			missingIndices = append(missingIndices, o.indices...)
			lastErr = o.err
			continue
		}
		globalPredictions = append(globalPredictions, o.preds...)
	}

	if len(globalPredictions) == 0 {
		return PredictResult{}, fmt.Errorf("prediction failed: no workers returned valid results: %w", lastErr)
	}

	log.Printf("[Orchestrator] Collected %d total tree predictions. Aggregating globally...", len(globalPredictions))

	var warning string
	if len(missingIndices) > 0 {
		sort.Slice(missingIndices, func(i, j int) bool { return missingIndices[i] < missingIndices[j] })
		warning = fmt.Sprintf("prediction computed using %d of %d trees; trees %v could not be retrieved (worker failures exhausted retries): %v",
			len(globalPredictions), meta.NEstimators, missingIndices, lastErr)
		log.Printf("[Orchestrator] %s", warning)
	}

	// AGGREGATION
	if taskType == "regression" {
		values := make([]float64, len(globalPredictions))
		for i, pred := range globalPredictions {
			values[i] = pred.Value
		}
		return PredictResult{Value: aggregateRegression(values), Warning: warning}, nil
	}
	return PredictResult{Value: aggregateClassification(globalPredictions), Warning: warning}, nil
}

// predictChunkWithRetry asks worker for the predictions of chunk. When the
// call fails, the whole chunk is reassigned to another healthy worker, at
// most MaxRetriesPerTree times.
func (p *WorkerPool) predictChunkWithRetry(ctx context.Context, req *pb.PredictRequest, chunk []int32, worker *WorkerClient) ([]*pb.TreePrediction, error) {
	for attempt := 1; ; attempt++ {
		workerReq := &pb.PredictRequest{
			ModelId:     req.ModelId,
			Features:    req.Features,
			TreeIndices: chunk,
		}

		resp, err := worker.Client.Predict(ctx, workerReq)
		if err == nil {
			return resp.Predictions, nil
		}
		log.Printf("[Orchestrator] Worker %s failed predict on model %s: %v", worker.Address, req.ModelId, err)

		if attempt > p.MaxRetriesPerTree {
			return nil, fmt.Errorf("trees %v of model %s failed %d times, giving up: %v", chunk, req.ModelId, attempt, err)
		}

		replacement, err := p.waitForReplacement(ctx, worker)
		if err != nil {
			return nil, fmt.Errorf("trees %v of model %s could not be reassigned: %w", chunk, req.ModelId, err)
		}
		log.Printf("[Orchestrator] Reassigning trees %v of model %s to worker %s.", chunk, req.ModelId, replacement.Address)
		worker = replacement
	}
}

// --------------------------- Aggregation Strategies ---------------------

// aggregateRegression calculates the mean of the results
func aggregateRegression(values []float64) string {
	if len(values) == 0 {
		return "0"
	}

	var sum float64
	for _, v := range values {
		sum += v
	}

	mean := sum / float64(len(values))
	return fmt.Sprintf("%f", mean)
}

// aggregateClassification does soft voting: class probabilities are summed
// over all trees (a class a tree never saw contributes 0) and the class with
// the highest total wins. Ties go to the first class in sorted order, as in
// scikit-learn's argmax.
func aggregateClassification(predictions []*pb.TreePrediction) string {
	sums := make(map[string]float64)

	for _, pred := range predictions {
		for i, class := range pred.Classes {
			if i < len(pred.Probabilities) {
				sums[class] += pred.Probabilities[i]
			}
		}
	}

	classes := make([]string, 0, len(sums))
	for class := range sums {
		classes = append(classes, class)
	}
	sort.Strings(classes)

	var best string
	bestSum := -1.0
	for _, class := range classes {
		if sums[class] > bestSum {
			bestSum = sums[class]
			best = class
		}
	}

	return best
}

// Close closes all connections
func (p *WorkerPool) Close() {
	for _, w := range p.Workers {
		w.Conn.Close()
	}
}
