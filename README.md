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
├── api/                     # Definizioni delle API (Protobuf)
│   └── proto/
│       └── worker/
│           └── v1/
│               └── worker.proto   # Definizione del servizio gRPC per i worker
│
├── cmd/                     # Entrypoint delle applicazioni (eseguibili)
│   ├── master/              # Codice per l'eseguibile del Master
│   │   └── main.go
│   └── worker/              # Codice per l'eseguibile del Worker
│       └── main.py
│
├── internal/                # Codice Go privato del Master (non esportabile)
│   ├── api/                 # Handler per le API REST
│   ├── config/              # Gestione della configurazione
│   ├── orchestrator/        # Logica di orchestrazione dei worker
│   └── platform/            # Interfacce verso servizi esterni (es. S3, gRPC client)
│
│
├── services/                # Codice Python per i servizi (in questo caso, il worker)
│   └── worker/
│       ├── ml/              # Logica di Machine Learning
│       │   ├── model.py     # Classe/funzioni per addestrare/predire
│       │   └── preprocessor.py # (Opzionale) Preprocessing dati
│       └── server.py        # Implementazione del server gRPC del worker
│
├── configs/                 # File di configurazione (es. config.yaml)
│
├── deployments/             # Manifesti per il deployment
│   └── docker-compose.yml   # Per l'ambiente locale
│
├── scripts/                 # Script di utilità (es. build.sh, run.sh)
│
├── Dockerfile.master        # Dockerfile per il Master Go
├── Dockerfile.worker        # Dockerfile per il Worker Python
├── go.mod                   # Gestione dipendenze Go
├── go.sum
├── requirements.txt         # Gestione dipendenze Python
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

## 3. Update/install the necessary Python libs
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
python cmd/worker/worker_main.py
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
```


## TEMP MINIO
```bash
docker-compose -f deployments/docker-compose.yml up -d
http://localhost:9001
```