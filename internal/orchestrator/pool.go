package orchestrator

// Package orchestrator allows for an Orchestration Approach (centralized approach)
// Master uses Orchestrator to manage and coordinate a pool of Workers

import (
	"context"
	"fmt"
	"log"
	"os/exec"
	"strconv"
	"sync"
	"time"

	pb "github.com/bytelisa/distributed-random-forest/api/proto/worker/v1"
	"github.com/bytelisa/distributed-random-forest/internal/config"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

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
}

// NewWorkerPool initializes connections to all workers listed in the config
func NewWorkerPool(addresses []string, healthTimeout int) (*WorkerPool, error) {
	pool := &WorkerPool{
		Workers:            make([]*WorkerClient, 0, len(addresses)),
		HealthCheckTimeout: time.Duration(healthTimeout) * time.Second,
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

// TrainDistributed splits the work among available (healthy) workers only and waits for completion
func (p *WorkerPool) TrainDistributed(ctx context.Context, req *pb.TrainRequest, storageCfg *config.StorageConfig) (*pb.TrainResponse, error) {

	activeWorkers, err := p.getHealthyWorkers(ctx)
	if err != nil {
		return nil, fmt.Errorf("training failed: %w", err)
	}
	numWorkers := len(activeWorkers)

	log.Printf("[Orchestrator] Running partitioner script for model %s into %d parts...", req.ModelId, numWorkers)

	// 1. RUN DATASET PARTITIONER
	// Execute the python script to prepare data on S3
	cmd := exec.CommandContext(ctx, "python", "scripts/partitioner.py",
		"--s3-endpoint", storageCfg.Endpoint,
		"--s3-access-key", storageCfg.AccessKey,
		"--s3-secret-key", storageCfg.SecretKey,
		"--s3-bucket", storageCfg.Bucket,
		"--source-key", req.DatasetUrl, // e.g., "data/iris.csv" (assuming we clean the s3:// prefix before)
		"--model-id", req.ModelId,
		"--num-partitions", fmt.Sprintf("%d", numWorkers),
	)

	if output, err := cmd.CombinedOutput(); err != nil {
		return nil, fmt.Errorf("partitioning failed: %w, logs: %s", err, string(output))
	}

	// DEBUG
	log.Printf("[Orchestrator] Partitioner script finished successfully.")

	// 2. DISTRIBUTE TRAINING TASKS

	// DEBUG
	datasetFolder := fmt.Sprintf("models/%s/dataset_partitions/", req.ModelId)
	log.Printf("[Orchestrator] Distributing training tasks. Dataset folder: %s", datasetFolder)

	var wg sync.WaitGroup
	errChan := make(chan error, numWorkers)

	for i, worker := range activeWorkers {
		wg.Add(1)
		go func(w *WorkerClient, idx int) {
			defer wg.Done()

			// We pass the base folder, not the specific file
			// The worker will figure out which file to grab
			datasetFolder := fmt.Sprintf("models/%s/dataset_partitions/", req.ModelId)

			//DEBUG
			log.Printf("[Orchestrator] Sending Train request to Worker %d: %d trees on dataset folder.", idx, req.NEstimators)

			workerReq := &pb.TrainRequest{
				ModelId:      req.ModelId,
				DatasetUrl:   datasetFolder,
				TaskType:     req.TaskType,
				TargetColumn: req.TargetColumn,
				NEstimators:  req.NEstimators, // Each worker trains a FULL forest (e.g., 100 trees)
				WorkerIndex:  int32(idx),
				TotalWorkers: int32(numWorkers),
			}

			resp, err := w.Client.Train(ctx, workerReq)
			if err != nil || !resp.Success {
				errChan <- fmt.Errorf("worker %d failed: %v", idx, err)
			}
		}(worker, i)
	}

	wg.Wait()
	close(errChan)

	// Check if any worker failed during execution
	// todo more fault tolerance here?
	if len(errChan) > 0 {
		err := <-errChan
		return &pb.TrainResponse{
			Success: false,
			Message: fmt.Sprintf("Distributed training failed mid-process. Error: %v", err),
		}, nil
	}

	return &pb.TrainResponse{
		Success: true,
		Message: fmt.Sprintf("Training completed successfully on %d workers.", numWorkers),
	}, nil
}

// THOUGHTS:
// Call Predict goroutines on the (correct!) workers
// Note: should make sure that we ask to predict to the workers who actually trained the model?
// Or maybe it's not relevant because the trained model is available on shared storage so any worker can use any part to contribute to the prediction?
// YES: final decision went on stateless workers, so the model "parts" (trained trees) are partitioned between the workers who simply download them.
// A worker can use trees he didn't train for inference purposes
// This also solves (partly) fault tolerance --> no lost state (no cached trained trees)

// PredictDistributed is responsible for the aggregation of inference results coming from the workers (Bagging - Aggregation Phase)
// taskType should be "classification" or "regression"
func (p *WorkerPool) PredictDistributed(ctx context.Context, req *pb.PredictRequest, taskType string) (string, error) {

	// DYNAMIC HEALTH CHECK
	// Needed to compute the correct indices and distribute work accordingly
	activeWorkers, err := p.getHealthyWorkers(ctx)
	if err != nil {
		return "", err
	}

	numWorkers := len(activeWorkers)
	log.Printf("[Orchestrator] Active workers for inference: %d (configured: %d)", numWorkers, len(p.Workers))

	resultsChan := make(chan []string, numWorkers)
	var wg sync.WaitGroup

	log.Printf("[Orchestrator] Broadcasting prediction request to %d workers...", numWorkers)

	// Loop on active workers
	for i, worker := range activeWorkers { // 'i' goes from 0 to (numWorkers-1)
		wg.Add(1)
		go func(w *WorkerClient, idx int) {
			defer wg.Done()

			workerReq := &pb.PredictRequest{
				ModelId:      req.ModelId,
				Features:     req.Features,
				WorkerIndex:  int32(idx),
				TotalWorkers: int32(numWorkers), // alive workers
			}

			resp, err := w.Client.Predict(ctx, workerReq)
			if err != nil {
				// todo manage fault tolerance here: what happens if a worker fails after it gets assigned a model part?
				log.Printf("[Orchestrator] Warning: Worker %s failed predict: %v", w.Address, err)
				return
			}

			if len(resp.Predictions) > 0 {
				resultsChan <- resp.Predictions
			}
		}(worker, i)
	}

	go func() {
		wg.Wait()
		close(resultsChan)
	}()

	// COLLECT PREDICTIONS
	// We merge all partial lists into one global list of votes/values
	// Note: Only aggregate once (no local aggregation on worker) in order to introduce less error.
	var globalPredictions []string

	for partialList := range resultsChan {
		globalPredictions = append(globalPredictions, partialList...)
	}

	if len(globalPredictions) == 0 {
		return "", fmt.Errorf("prediction failed: no workers returned valid results")
	}

	log.Printf("[Orchestrator] Collected %d total tree predictions. Aggregating globally...", len(globalPredictions))

	// AGGREGATION
	if taskType == "regression" {
		return aggregateRegression(globalPredictions), nil
	} else {
		return aggregateClassification(globalPredictions), nil
	}
}

// --------------------------- Aggregation Strategies ---------------------

// aggregateRegression calculates the mean of the results
func aggregateRegression(results []string) string {
	var sum float64
	count := 0

	for _, r := range results {
		val, err := strconv.ParseFloat(r, 64)
		if err == nil {
			sum += val
			count++
		} else {
			log.Printf("[Orchestrator] Error parsing float result: %s", r)
		}
	}

	if count == 0 {
		return "0"
	}

	mean := sum / float64(count)
	return fmt.Sprintf("%f", mean)
}

// aggregateClassification calculates the mode (majority vote)
func aggregateClassification(results []string) string {
	counts := make(map[string]int)

	for _, r := range results {
		counts[r]++
	}

	var bestVal string
	maxCount := -1

	for val, c := range counts {
		if c > maxCount {
			maxCount = c
			bestVal = val
		}
	}

	return bestVal
}

// Close closes all connections
func (p *WorkerPool) Close() {
	for _, w := range p.Workers {
		w.Conn.Close()
	}
}
