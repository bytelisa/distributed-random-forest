# A Distributed RandomForest Model

## Prerequisites

- Go 1.25+ installed
- Python 3.9+ installed
- `protoc` (Protocol Buffers compiler) installed
- `protoc-gen-go` and `protoc-gen-go-grpc` plugins installed
- Docker + Docker Compose (for the local MinIO setup below)

## Project structure

```text
distributed-random-forest/
├── api/                     # API definitions (Protobuf)
│   └── proto/
│       └── worker/
│           └── v1/
│               └── worker.proto   # Interface of the gRPC services of the worker
│
├── cmd/                     # Application entrypoints (exe)
│   ├── master/              # Master main
│   │   └── master_main.go
│   └── worker/              # Worker main
│       └── worker_main.py
│
├── internal/                # Private Go code of the Master
│   ├── api/                 # REST API handler
│   ├── config/              # Configuration management
│   └── orchestrator/        # Worker orchestration logic, S3 access, crash recovery
│
├── services/                # Python code of the workers
│   └── worker/
│       ├── ml/
│       │   └── model.py     # ML model: training and inference functions
│       ├── platform/
│       │   └── storage.py   # S3 client
│       └── worker_service.py # Implementation of the gRPC server of the Worker
│
├── scripts/
│   └── partitioner.py       # Dataset partitioning, invoked by the master
│
├── configs/                 # Configuration files
│   ├── config.yaml          # Local, non-Docker execution
│   ├── config.docker.yaml   # Docker Compose (local, MinIO)
│   └── config.aws.yaml      # AWS EC2 execution (real S3)
│
├── docker-compose.yml       # Local deployment (MinIO + master + workers)
├── Dockerfile.master        # Dockerfile for the Go Master
├── Dockerfile.worker        # Dockerfile for the Python Worker
├── go.mod                   # Go dependencies
├── go.sum
├── requirements.txt         # Python dependencies
└── README.md
```

## 1. Set up the virtual environment for Python
```bash
python -m venv venv
.\venv\Scripts\activate
```

## 2. Install Python requirements
```bash
pip install -r requirements.txt
```

## 3. Generate .go files using protoc
```bash
protoc --go_out=. --go_opt=paths=source_relative --go-grpc_out=. --go-grpc_opt=paths=source_relative api/proto/worker/v1/worker.proto
```

## 4. Generate .py files
```bash
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. api/proto/worker/v1/worker.proto
```

## 5. Launch a Worker
```bash
python cmd/worker/worker_main.py --port 50051
```

## 6. Launch the Master
```bash
go run cmd/master/master_main.go
```

## Local storage: MinIO instead of S3
```bash
docker-compose up -d
```
Manage the bucket and upload datasets (e.g. `data/iris.csv`, `data/housing.csv`) on this link:
```bash
http://localhost:9001
```
(login `minioadmin` / `minioadmin`)

## 7. Send an example training request
`dataset_url` is a **bare object key** inside the bucket (not a full `s3://...` URL) — make sure the file has already been uploaded to MinIO first.

Example for a classification task:
```bash
curl.exe -X POST http://localhost:8080/train -H "Content-Type: application/json" -d '{\"dataset_url\": \"iris.csv\", \"task_type\": \"classification\", \"target_column\": \"Species\", \"n_estimators\": 10}'
```
Example for a regression task:
```bash
curl.exe -X POST http://localhost:8080/train -H "Content-Type: application/json" -d '{\"dataset_url\": \"housing.csv\", \"task_type\": \"regression\", \"target_column\": \"median_house_value\", \"n_estimators\": 10}'
```
The response is immediate (`202 Accepted`) and only contains the `model_id` — training keeps running in the background:
```json
{"model_id": "...", "status": "training", "message": "Training started."}
```

## 8. Poll the model's status
```bash
curl.exe http://localhost:8080/models/<MODEL_ID>
```
Wait until the status is `"ready"` before running inference.

## 9. Send an example predict request
Example for a classification task:
```bash
curl.exe -X POST http://localhost:8080/predict/<MODEL_ID> -H "Content-Type: application/json" -d '{\"features\": [5.0, 3.6, 1.4, 0.2], \"task_type\": \"classification\"}'
```
Example for a regression task:
```bash
curl -X POST http://localhost:8080/predict/<MODEL_ID> -H "Content-Type: application/json" -d '{"features": [-122.23, 37.88, 41.0, 880.0, 129.0, 322.0, 126.0, 8.32], "task_type": "regression"}'
```

## 10. Test orchestration
Build the images:
```bash
docker-compose up --build
```
or just launch the containers:
```bash
docker-compose up
```
or rebuild a single image:
```bash
docker-compose up --build -d master
```

## 11. Send training request from file
```bash
curl.exe -X POST http://localhost:8080/train -H "Content-Type: application/json" -d "@train_request.json"
```

## 12. Send predict request from file
```bash
curl.exe -X POST http://localhost:8080/predict/<MODEL_ID> -H "Content-Type: application/json" -d "@predict_request.json"
```
