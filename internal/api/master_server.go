package api

// Master Server: receives and handles HTTP requests coming from users

import (
	"context"
	"fmt"
	"net/http"
	"time"

	pb "github.com/bytelisa/distributed-random-forest/api/proto/worker/v1"
	"github.com/bytelisa/distributed-random-forest/internal/config"
	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

// Server holds the dependencies for the HTTP handlers
type Server struct {
	router     *gin.Engine
	config     *config.Config
	grpcClient pb.WorkerClient // this will later become a pool of clients
}

// NewServer initializes the REST API server
func NewServer(cfg *config.Config) (*Server, error) {
	// 1. Initialize gRPC connection
	workerAddr := cfg.Workers.Addresses[0]
	conn, err := grpc.NewClient(workerAddr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		return nil, fmt.Errorf("failed to connect to worker: %w", err)
	}
	client := pb.NewWorkerClient(conn)

	// 2. Setup Router
	router := gin.Default()

	s := &Server{
		router:     router,
		config:     cfg,
		grpcClient: client,
	}

	// 3. Define Routes
	router.POST("/train", s.handleTrain)
	router.POST("/predict/:model_id", s.handlePredict)

	return s, nil
}

// Start runs the HTTP server
func (s *Server) Start(addr string) error {
	return s.router.Run(addr)
}

// handleTrain processes the training request
func (s *Server) handleTrain(c *gin.Context) {
	var req TrainRequest
	// Bind JSON body to struct
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	// Generate a unique Model ID
	modelID := uuid.New().String()

	// Map generic task type string to Protobuf enum
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

	// Prepare gRPC request
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	grpcReq := &pb.TrainRequest{
		ModelId:      modelID,
		DatasetUrl:   req.DatasetURL,
		TaskType:     pbTaskType,
		TargetColumn: req.TargetColumn,
		NEstimators:  int32(req.NEstimators),
	}

	// Call Worker via gRPC (Blocking call)
	grpcResp, err := s.grpcClient.Train(ctx, grpcReq)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Worker training failed: " + err.Error()})
		return
	}

	// UPDATE: Check success from worker response
	status := "completed"
	if !grpcResp.Success {
		status = "failed"
		// If worker explicitly says success=false, we might want to return 500 or 400
		c.JSON(http.StatusInternalServerError, gin.H{"error": grpcResp.Message})
		return
	}

	fmt.Printf("[Master] Training status: %s\n", status)
	// UPDATE: Return 200 OK with the actual message from the worker
	c.JSON(http.StatusOK, TrainResponse{
		ModelID: modelID,
		Status:  status,
		Message: grpcResp.Message, // <--- This now comes from the Python Worker!
	})
}

// handlePredict processes the inference request
func (s *Server) handlePredict(c *gin.Context) {
	modelID := c.Param("model_id")

	var req PredictRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	// Call Worker via gRPC
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	grpcReq := &pb.PredictRequest{
		ModelId:  modelID,
		Features: req.Features,
	}

	resp, err := s.grpcClient.Predict(ctx, grpcReq)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Inference failed: " + err.Error()})
		return
	}

	c.JSON(http.StatusOK, PredictResponse{
		ModelID:    modelID,
		Prediction: resp.Prediction,
	})
}
