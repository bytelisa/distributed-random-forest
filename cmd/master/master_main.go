package main

import (
	"context"
	"fmt"
	"log"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"

	// Import the generated packet.
	pb "github.com/bytelisa/distributed-random-forest/api/proto/worker/v1"
)

func main() {
	fmt.Println("Master service starting...")

	// The server acts as a grpc client towards the worker, which acts as a grpc server, exporting its services.

	// Connect to Worker (assume localhost:50051 but todo fix)
	conn, err := grpc.NewClient("localhost:50051", grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		log.Fatalf("could not connect: %v", err)
	}
	defer conn.Close()

	// Create Client Stub
	client := pb.NewWorkerClient(conn)

	// Call HealthCheck just for a quick test
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()

	fmt.Printf("[Master] Ping worker...")
	r, err := client.Health(ctx, &pb.HealthRequest{})
	if err != nil {
		log.Fatalf("could not health check: %v", err)
	}
	fmt.Printf("Worker Response: Healthy=%v\n", r.Healthy)

	// Test with fake Training Request
	fmt.Println("Sending Training request...")
	trainResp, err := client.Train(context.Background(), &pb.TrainRequest{
		ModelId:     "test-model-uuid",
		DatasetUrl:  "data/iris.csv",
		TaskType:    pb.TaskType_CLASSIFICATION_TASK,
		NEstimators: 10,
	})
	if err != nil {
		log.Fatalf("Training failed: %v", err)
	}
	fmt.Printf("Training Response: %s\n", trainResp.Message)

}
