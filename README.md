# A Distributed RandomForest Model

## Prerequisites

- Go 1.20+ installed
- Python 3.9+ installed
- `protoc` (Protocol Buffers compiler) installed
- `protoc-gen-go` and `protoc-gen-go-grpc` plugins installed
- Python Community Edition plugin installed (optional)

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
│   │   └── main.go
│   └── worker/              # Worker main
│       └── main.py
│
├── internal/                # Private Go code of the Master 
│   ├── api/                 # REST API handler
│   ├── config/              # Configuration management
│   ├── orchestrator/        # Worker Orchestration logic
│   └── platform/            # Interfaces to external services (eg. S3, gRPC client)
│
│
├── services/                # Python code of the workers
│   └── worker/
│       ├── ml/              # Machine Learning logic
│       │   ├── model.py     # ML model: training and inference functions
│       │   └── preprocessor.py # (Later) Data preprocessing 
│       └── server.py        # Implementation of the gRPC server of the Worker
│
├── configs/                 # File di configurazione (es. config.yaml)
│
├── deployments/             # Deployment
│   └── docker-compose.yml   # Local deployment
│
├── scripts/                 # Utility scripts (eg. build.sh, run.sh)
│
├── Dockerfile.master        # Dockerfile for the Go Master
├── Dockerfile.worker        # Dockerfile for the Python Worker
├── go.mod                   # Go dependencies
├── go.sum
├── requirements.txt         # Python dependencies
└── README.md
```
## 1. Create `go.mod`

```bash
go mod init distributed-random-forest
go mod tidy
```

## 2. Activate the virtual environment for Python
```bash
.\venv\Scripts\activate
```

## 3. Update/install Python requirements
```bash
pip install grpcio grpcio-tools protobuf scikit-learn pandas numpy
```

## 4. Generate .go files using protoc 
```bash
protoc --go_out=. --go_opt=paths=source_relative --go-grpc_out=. --go-grpc_opt=paths=source_relative api/proto/worker/v1/worker.proto
```

## 5. Generate .py files 
```bash
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. api/proto/worker/v1/worker.proto
```

## 6. Launch a Worker
```bash
python cmd/worker/worker_main.py --port 50051
```

## 7. Launch the Master
```bash
go run cmd/master/master_main.go
```

## 8. Send an example training request
```bash
curl.exe -X POST http://localhost:8080/train -H "Content-Type: application/json" -d '{\"dataset_url\": \"s3://example-storage/iris.csv\", \"task_type\": \"classification\", \"target_column\": \"Species\", \"n_estimators\": 10}'
``````

## 9. Send an example predict request
```bash
curl.exe -X POST http://localhost:8080/predict/test-model-iris -H "Content-Type: application/json" -d '{\"features\": [5.0, 3.6, 1.4, 0.2]}'
curl.exe -X POST http://localhost:8080/predict/b6d09e79-b286-4e4d-b978-efe872fda83c -H "Content-Type: application/json" -d '{\"features\": [5.0, 3.6, 1.4, 0.2]}'
```


## TEMP MINIO
```bash
docker-compose -f deployments/docker-compose.yml up -d
http://localhost:9001
```