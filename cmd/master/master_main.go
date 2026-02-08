// cmd/master/main.go
package main

import (
	"fmt"
	"log"

	"github.com/bytelisa/distributed-random-forest/internal/api"
	"github.com/bytelisa/distributed-random-forest/internal/config"
)

func main() {
	// Load Configuration file
	cfg, err := config.LoadConfig()
	if err != nil {
		log.Fatalf("Failed to load config: %v", err)
	}

	// Initialize REST Server
	server, err := api.NewServer(cfg)
	if err != nil {
		log.Fatalf("Failed to create server: %v", err)
	}

	// Start Server
	serverAddr := fmt.Sprintf(":%d", cfg.Server.Port)
	fmt.Printf("[Master] REST API listening on %s\n", serverAddr)

	if err := server.Start(serverAddr); err != nil {
		log.Fatalf("Server failed: %v", err)
	}
}
