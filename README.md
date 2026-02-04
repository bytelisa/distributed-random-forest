# A Distributed RandomForest Model

## Prerequisites

- Go 1.20+ installed
- `protoc` (Protocol Buffers compiler) installed
- `protoc-gen-go` and `protoc-gen-go-grpc` plugins installed
- Update HERE

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
go mod init FIX HERE
go mod tidy
```