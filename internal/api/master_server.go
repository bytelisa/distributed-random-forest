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

	// 2. Setup Router
	router := gin.Default()

	s := &Server{
		router:     router,
		config:     cfg,
		workerPool: pool,
	}

	// Routes
	router.POST("/train", s.handleTrain)
	router.POST("/predict/:model_id", s.handlePredict)

	return s, nil
}

// Start runs the HTTP server
func (s *Server) Start(addr string) error {
	return s.router.Run(addr)
}

func (s *Server) handleTrain(c *gin.Context) {
	var req TrainRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	modelID := uuid.New().String()

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

	// Read timeout from config file
	timeout := time.Duration(s.config.System.TimeoutTraining) * time.Second
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	grpcReq := &pb.TrainRequest{
		ModelId:      modelID,
		DatasetUrl:   req.DatasetURL,
		TaskType:     pbTaskType,
		TargetColumn: req.TargetColumn,
		NEstimators:  int32(req.NEstimators),
	}

	// Call Distributed Training
	orchestratorResp, err := s.workerPool.TrainDistributed(ctx, grpcReq, &s.config.Storage)

	if err != nil {
		// System error (e.g. no workers)
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	if !orchestratorResp.Success {
		// Logic error (one worker failed)
		c.JSON(http.StatusInternalServerError, gin.H{"error": orchestratorResp.Message})
		return
	}

	c.JSON(http.StatusOK, TrainResponse{
		ModelID: modelID,
		Status:  "completed",
		Message: orchestratorResp.Message,
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
