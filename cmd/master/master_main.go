package main

import (
	"fmt"
	"log"

	"github.com/bytelisa/distributed-random-forest/internal/api"
	"github.com/bytelisa/distributed-random-forest/internal/config"
)

func main() {
	// Upload config (server port, worker address, s3 creds...)
	cfg, err := config.LoadConfig()
	if err != nil {
		log.Fatalf("[Master] Failed to load config: %v", err)
	}

	// Initialize REST server (he'll open the grpc connection to the workers)
	server, err := api.NewServer(cfg)
	if err != nil {
		log.Fatalf("[Master] Failed to create server: %v", err)
	}

	// Run Server and wait for requests
	serverAddr := fmt.Sprintf(":%d", cfg.Server.Port)
	fmt.Printf("[Master] REST API listening on %s... (Waiting for requests)\n", serverAddr)

	if err := server.Start(serverAddr); err != nil {
		log.Fatalf("[Master] Server failed: %v", err)
	}
}
