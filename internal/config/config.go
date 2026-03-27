package config

import (
	"log"
	"strings"

	"github.com/spf13/viper"
)

// This file reads and allows you to interact with your configuration file.

type Config struct {
	App     AppConfig     `mapstructure:"app"`
	Server  ServerConfig  `mapstructure:"server"`
	Workers WorkerConfig  `mapstructure:"workers"`
	Storage StorageConfig `mapstructure:"storage"`
	Tasks   []TaskConfig  `mapstructure:"tasks"`
	System  SystemConfig  `mapstructure:"system"`
}

type AppConfig struct {
	Env string `mapstructure:"env"`
}

type ServerConfig struct {
	Port int `mapstructure:"port"`
}

type WorkerConfig struct {
	Addresses []string `mapstructure:"addresses"`
}

type StorageConfig struct {
	Type      string `mapstructure:"type"`
	Endpoint  string `mapstructure:"endpoint"`   // NEW
	AccessKey string `mapstructure:"access_key"` // NEW
	SecretKey string `mapstructure:"secret_key"` // NEW
	Bucket    string `mapstructure:"bucket"`     // NEW

	BasePath string `mapstructure:"base_path"`
	// LocalTempDir is mainly used by workers, but good to map it anyway
	LocalTempDir string `mapstructure:"local_temp_dir"`
}

type TaskConfig struct {
	Name            string         `mapstructure:"name"`
	Type            string         `mapstructure:"type"`
	DatasetPath     string         `mapstructure:"dataset_path"`
	TargetColumn    string         `mapstructure:"target_column"`
	Hyperparameters map[string]int `mapstructure:"hyperparameters"` // Semplificato a int per ora
	TestFeatures    []float32      `mapstructure:"test_features"`
}

type SystemConfig struct {
	TimeoutTraining    int `mapstructure:"timeout_training_seconds"`
	TimeoutPrediction  int `mapstructure:"timeout_prediction_seconds"`
	TimeoutHealthCheck int `mapstructure:"timeout_health_check_seconds"`
}

// LoadConfig reads the config file
func LoadConfig() (*Config, error) {
	v := viper.New()

	// Filename w/o extension
	v.SetConfigName("config")
	// Filetype
	v.SetConfigType("yaml")
	// Paths
	v.AddConfigPath("./configs")
	v.AddConfigPath(".")

	// Enable reading of env vars
	// Es: SERVER_PORT will overwrite server.port
	v.SetEnvKeyReplacer(strings.NewReplacer(".", "_"))
	v.AutomaticEnv()

	if err := v.ReadInConfig(); err != nil {
		return nil, err
	}

	var cfg Config
	if err := v.Unmarshal(&cfg); err != nil {
		return nil, err
	}

	log.Printf("Configuration loaded with success from: %s", v.ConfigFileUsed())
	return &cfg, nil
}
