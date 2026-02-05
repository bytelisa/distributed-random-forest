
# the main turns on the server and manages the connection

import logging
import sys
import os
from concurrent import futures
import grpc

sys.path.append(os.getcwd())

from api.proto.worker.v1 import worker_pb2_grpc
from services.worker.worker_service import WorkerService

def serve():
    print("Worker service starting...")

    # The worker acts as a grpc server towards the master, which acts as a grpc client requesting the worker's services.

    # Create a gRPC sever
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

    # Register WorkerService on the server
    worker_pb2_grpc.add_WorkerServiceServicer_to_server(WorkerService(), server)

    # Listen on port 50051
    port = "50051"
    server.add_insecure_port(f'[::]:{port}')
    print(f"Worker server listening on port {port}...")

    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    logging.basicConfig()
    serve()