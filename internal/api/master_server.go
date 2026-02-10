package api

import (
	"context"
	"fmt"
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
	workerPool *orchestrator.WorkerPool // Usiamo il Pool, non più il singolo grpcClient
}

// NewServer initializes the REST API server
func NewServer(cfg *config.Config) (*Server, error) {
	// 1. Initialize Worker Pool (QUESTA È LA PARTE CHE MANCAVA O ERA VECCHIA)
	// Questo chiamerà la funzione in pool.go che stampa "[Orchestrator] Connected..."
	pool, err := orchestrator.NewWorkerPool(cfg.Workers.Addresses)
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

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	grpcReq := &pb.TrainRequest{
		ModelId:      modelID,
		DatasetUrl:   req.DatasetURL,
		TaskType:     pbTaskType,
		TargetColumn: req.TargetColumn,
		NEstimators:  int32(req.NEstimators),
	}

	// UPDATE: Call Distributed Training
	orchestratorResp, err := s.workerPool.TrainDistributed(ctx, grpcReq)

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

func (s *Server) handlePredict(c *gin.Context) {
	modelID := c.Param("model_id")

	var req PredictRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	grpcReq := &pb.PredictRequest{
		ModelId:  modelID,
		Features: req.Features,
	}

	// --- TEMPORANEO: usa solo il primo worker ---
	if len(s.workerPool.Workers) == 0 {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "No workers available"})
		return
	}
	worker := s.workerPool.Workers[0]
	// -----------------------------------------------

	resp, err := worker.Client.Predict(ctx, grpcReq)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Inference failed: " + err.Error()})
		return
	}

	c.JSON(http.StatusOK, PredictResponse{
		ModelID:    modelID,
		Prediction: resp.Prediction,
	})
}
