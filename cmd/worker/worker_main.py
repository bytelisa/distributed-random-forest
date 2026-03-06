
# the main turns on the server and manages the connection

import logging
import sys
import os
from concurrent import futures
import grpc
import argparse # for 'port' argument

sys.path.append(os.getcwd())

from api.proto.worker.v1 import worker_pb2_grpc
from services.worker.worker_service import WorkerService
from services.worker.config import load_config

def serve():
    print("Worker service starting...")

    try:
        cfg = load_config()
        print(f"[Worker] Config loaded. Max threads: {cfg.grpc_max_threads}")
    except Exception as e:
        print(f"[Worker] Failed to load config: {e}")
        return

    # The worker acts as a grpc server towards the master, which acts as a grpc client requesting the worker's services.

    # Parse command line arguments to launch worker on specified port
    parser = argparse.ArgumentParser(description='Distributed Random Forest Worker')
    parser.add_argument('--port', type=str, default='50051', help='Port to listen on')
    args = parser.parse_args()

    port = int(args.port)

    # Create a gRPC server
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=cfg.grpc_max_threads))

    # Register WorkerService on the server
    worker_pb2_grpc.add_WorkerServicer_to_server(WorkerService(worker_id=port), server)

    # Start listening on port
    server.add_insecure_port(f'[::]:{port}')
    print(f"[Worker] Server listening on port {port}...")

    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    logging.basicConfig()
    serve()