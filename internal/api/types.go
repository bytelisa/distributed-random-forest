package api

// This class basically translates user's JSON requests (see openapi.yaml) into Go

// TrainRequest represents the JSON body received in POST /train
type TrainRequest struct {
	DatasetURL   string         `json:"dataset_url" binding:"required"`
	TaskType     string         `json:"task_type" binding:"required,oneof=classification regression"`
	TargetColumn string         `json:"target_column" binding:"required"`
	NEstimators  int            `json:"n_estimators"` // Optional, default handled in logic
	Hyperparams  map[string]int `json:"hyperparameters"`
}

// TrainResponse represents the JSON response for POST /train
type TrainResponse struct {
	ModelID string `json:"model_id"`
	Status  string `json:"status"`
	Message string `json:"message"`
}

// PredictRequest represents the JSON body received in POST /predict
type PredictRequest struct {
	// The openapi.yaml allows generic objects, but for simplicity
	// let's assume the user sends a list of floats for now,
	// or we can parse a generic map if needed later.
	// Matching the gRPC logic:
	Features []float32 `json:"features" binding:"required"`
}

// PredictResponse represents the JSON response for POST /predict
type PredictResponse struct {
	ModelID    string `json:"model_id"`
	Prediction string `json:"prediction"`
}
