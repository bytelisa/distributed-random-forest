package main

import (
	"context"
	"fmt"
	"log"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"

	pb "github.com/bytelisa/distributed-random-forest/api/proto/worker/v1"
	"github.com/bytelisa/distributed-random-forest/internal/config"
)

func main() {
	fmt.Println("[Master] Starting...")

	// Load Configuration data
	cfg, err := config.LoadConfig()
	if err != nil {
		log.Fatalf("Errore nel caricamento della configurazione: %v", err)
	}

	// Get workers from config (just one for now, later will iterate on cfg.Workers.Addresses)
	workerAddr := cfg.Workers.Addresses[0]
	fmt.Printf("[Master] Connecting to worker at %s...\n", workerAddr)

	conn, err := grpc.NewClient(workerAddr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		log.Fatalf("did not connect: %v", err)
	}
	defer conn.Close()

	client := pb.NewWorkerClient(conn)

	// ------------------  HEALTH CHECK  ------------------
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()

	healthResp, err := client.Health(ctx, &pb.HealthRequest{})
	if err != nil {
		log.Fatalf("Health check failed: %v", err)
	}
	fmt.Printf("[Master] Worker Status: Healthy=%v\n", healthResp.Healthy)

	// Loop on tasks defined in the config file
	for _, task := range cfg.Tasks {
		fmt.Printf("\n--- PROCESSING TASK: %s ---\n", task.Name)

		// Determine task type
		var pbTaskType pb.TaskType
		if task.Type == "classification" {
			pbTaskType = pb.TaskType_CLASSIFICATION_TASK
		} else if task.Type == "regression" {
			pbTaskType = pb.TaskType_REGRESSION_TASK
		} else {
			log.Printf("Tipo task sconosciuto: %s, skipping...", task.Type)
			continue
		}

		// ------------------ TRAIN ------------------
		fmt.Printf(" -> Sending Train Request (Dataset: %s)...\n", task.DatasetPath)
		trainResp, err := client.Train(context.Background(), &pb.TrainRequest{
			ModelId:      "model-" + task.Name, // Generiamo un ID basato sul nome
			DatasetUrl:   task.DatasetPath,
			TaskType:     pbTaskType,
			NEstimators:  int32(task.Hyperparameters["n_estimators"]),
			TargetColumn: task.TargetColumn,
		})

		if err != nil {
			log.Printf("Training failed for %s: %v\n", task.Name, err)
			continue
		}
		fmt.Printf(" -> Training Response: %s\n", trainResp.Message)

		// ------------------ PREDICT ------------------
		if len(task.TestFeatures) > 0 {
			fmt.Printf(" -> Predicting with test features: %v\n", task.TestFeatures)
			predResp, err := client.Predict(context.Background(), &pb.PredictRequest{
				ModelId:  "model-" + task.Name,
				Features: task.TestFeatures,
			})

			if err != nil {
				log.Printf("Prediction failed for %s: %v\n", task.Name, err)
			} else {
				fmt.Printf(" -> Prediction Result: %s\n", predResp.Prediction)
			}
		}
	}

	fmt.Println("\n[Master] All tasks processed.")
}
