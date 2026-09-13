package api

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"time"

	pb "github.com/bytelisa/distributed-random-forest/api/proto/worker/v1"
	"github.com/bytelisa/distributed-random-forest/internal/config"
	"github.com/bytelisa/distributed-random-forest/internal/orchestrator" // Assicurati che l'import sia corretto
	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
)

// Server holds the dependencies
type Server struct {
	router     *gin.Engine
	config     *config.Config
	workerPool *orchestrator.WorkerPool // Uses a pool of workers
}

// NewServer initializes the REST API server
func NewServer(cfg *config.Config) (*Server, error) {

	// 1. Initialize Worker Pool
	pool, err := orchestrator.NewWorkerPool(cfg.Workers.Addresses, cfg.System.TimeoutHealthCheck)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize worker pool: %w", err)
	}

	// Cold standby recovery: if this instance was just launched by the Auto
	// Scaling Group after a previous master crashed, catch up on any
	// training left incomplete. Runs in the background so startup (and the
	// ALB health check) isn't blocked on it.
	go pool.ReconcileIncompleteTrainings(context.Background(), cfg)

	// 2. Setup Router
	router := gin.Default()

	s := &Server{
		router:     router,
		config:     cfg,
		workerPool: pool,
	}

	// Routes
	router.GET("/health", s.handleHealth)
	router.POST("/train", s.handleTrain)
	router.GET("/models/:model_id", s.handleModelStatus)
	router.POST("/predict/:model_id", s.handlePredict)

	return s, nil
}

// Start runs the HTTP server
func (s *Server) Start(addr string) error {
	return s.router.Run(addr)
}

// Router exposes the underlying HTTP handler, so tests can drive it directly
// (e.g. via httptest) without binding a real port.
func (s *Server) Router() http.Handler {
	return s.router
}

// WorkerPool exposes the underlying worker pool, so tests can substitute
// fault-injecting fakes (e.g. WorkerPool().PartitionDataset) before issuing
// requests against Router().
func (s *Server) WorkerPool() *orchestrator.WorkerPool {
	return s.workerPool
}

// handleHealth is used by the Application Load Balancer's target group to
// check whether this master instance is alive
func (s *Server) handleHealth(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{"status": "ok"})
}

func (s *Server) handleTrain(c *gin.Context) {
	var req TrainRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	var pbTaskType pb.TaskType
	switch req.TaskType {
	case "classification":
		pbTaskType = pb.TaskType_CLASSIFICATION_TASK
	case "regression":
		pbTaskType = pb.TaskType_REGRESSION_TASK
	default:
		c.JSON(http.StatusBadRequest, gin.H{"error": "invalid task_type"})
		return
	}

	modelID := uuid.New().String()

	grpcReq := &pb.TrainRequest{
		ModelId:      modelID,
		DatasetUrl:   req.DatasetURL,
		TaskType:     pbTaskType,
		TargetColumn: req.TargetColumn,
		NEstimators:  int32(req.NEstimators),
	}

	// Respond with the model_id right away: if the master crashes while
	// training is still running, the client has already learned the ID it
	// needs to check on later, instead of losing it along with the broken
	// connection. The Auto Scaling Group-recovered master picks up any
	// unfinished work on its own (see orchestrator.ReconcileIncompleteTrainings).
	c.JSON(http.StatusAccepted, TrainResponse{
		ModelID: modelID,
		Status:  "training",
		Message: "Training started.",
	})

	// The actual distributed training keeps running in the background
	go func() {
		timeout := time.Duration(s.config.System.TimeoutTraining) * time.Second
		ctx, cancel := context.WithTimeout(context.Background(), timeout)
		defer cancel()

		orchestratorResp, err := s.workerPool.TrainDistributed(ctx, grpcReq, &s.config.Storage)
		if err != nil {
			log.Printf("[Master] Training %s failed: %v", modelID, err)
			return
		}
		if !orchestratorResp.Success {
			log.Printf("[Master] Training %s failed: %s", modelID, orchestratorResp.Message)
			return
		}
		log.Printf("[Master] Training %s completed: %s", modelID, orchestratorResp.Message)
	}()
}

// handleModelStatus reports the status of a previously requested training:
// not found, still training, or ready for inference
func (s *Server) handleModelStatus(c *gin.Context) {
	modelID := c.Param("model_id")

	timeout := time.Duration(s.config.System.TimeoutPrediction) * time.Second
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	status, err := orchestrator.GetModelStatus(ctx, &s.config.Storage, modelID)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	if status == "not_found" {
		c.JSON(http.StatusNotFound, gin.H{"error": "model not found", "model_id": modelID})
		return
	}

	c.JSON(http.StatusOK, TrainResponse{
		ModelID: modelID,
		Status:  status,
	})
}

// handlePredict handles prediction requests coming from the HTTP client
func (s *Server) handlePredict(c *gin.Context) {
	modelID := c.Param("model_id")

	var req PredictRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	// Timeout for inference (read from config file)
	timeout := time.Duration(s.config.System.TimeoutPrediction) * time.Second
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	// Prepare base request
	grpcReq := &pb.PredictRequest{
		ModelId:  modelID,
		Features: req.Features,
	}

	// Send request to Orchestrator
	predictionResult, err := s.workerPool.PredictDistributed(ctx, grpcReq, req.TaskType)

	if err != nil {
		log.Printf("[Master] Inference error: %v", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Distributed inference failed: " + err.Error()})
		return
	}

	// Send response to Client
	c.JSON(http.StatusOK, PredictResponse{
		ModelID:    modelID,
		Prediction: predictionResult,
	})
}
