// Package testutil provides fakes used across the integration tests in
// test/orchestrator and test/api: an in-process gRPC worker
// (FakeWorker) and an in-memory S3-compatible HTTP server (FakeS3).
// Both exist so the fault-tolerance tests can exercise the real master
// code (WorkerPool, Server) without depending on a Python interpreter,
// scikit-learn, or a real MinIO/S3 endpoint.
package testutil

import (
	"context"
	"net"
	"sync"
	"testing"

	pb "github.com/bytelisa/distributed-random-forest/api/proto/worker/v1"
	"google.golang.org/grpc"
)

// FakeWorker is a configurable in-process gRPC worker. By default it
// reports healthy, and Train/Predict succeed trivially; tests override
// TrainFunc/PredictFunc/Healthy to inject specific fault-tolerance
// scenarios (crashes, delays, malformed responses).
type FakeWorker struct {
	pb.UnimplementedWorkerServer

	mu      sync.Mutex
	healthy bool

	// TrainFunc, if set, overrides the default (always-succeeds) Train
	// behavior.
	TrainFunc func(ctx context.Context, req *pb.TrainRequest) (*pb.TrainResponse, error)
	// PredictFunc, if set, overrides the default (one dummy vote per
	// requested tree) Predict behavior.
	PredictFunc func(ctx context.Context, req *pb.PredictRequest) (*pb.PredictResponse, error)

	trainCalls   []*pb.TrainRequest
	predictCalls []*pb.PredictRequest

	// Address is the loopback "host:port" the fake worker listens on -
	// what goes into config.WorkerConfig.Addresses in tests.
	Address string

	server *grpc.Server
}

// NewFakeWorker starts a fake worker on a loopback TCP port (needed
// because WorkerPool dials real addresses via grpc.NewClient) and stops it
// via t.Cleanup. Reports healthy by default.
func NewFakeWorker(t *testing.T) *FakeWorker {
	t.Helper()

	w := &FakeWorker{healthy: true}

	lis, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("testutil: failed to listen: %v", err)
	}

	srv := grpc.NewServer()
	pb.RegisterWorkerServer(srv, w)
	w.server = srv
	w.Address = lis.Addr().String()

	go func() {
		_ = srv.Serve(lis)
	}()

	t.Cleanup(srv.Stop)

	return w
}

// UnreachableAddress returns a loopback address nothing is listening on, to
// simulate a worker that is configured but unreachable (connection refused)
// at health-check time.
func UnreachableAddress(t *testing.T) string {
	t.Helper()

	lis, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("testutil: failed to reserve a port: %v", err)
	}
	addr := lis.Addr().String()
	_ = lis.Close()
	return addr
}

// SetHealthy toggles the worker's reported health, safe for concurrent use
// (e.g. flipping a worker "dead" mid-test).
func (w *FakeWorker) SetHealthy(healthy bool) {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.healthy = healthy
}

// TrainCalls returns a snapshot of every TrainRequest received so far.
func (w *FakeWorker) TrainCalls() []*pb.TrainRequest {
	w.mu.Lock()
	defer w.mu.Unlock()
	out := make([]*pb.TrainRequest, len(w.trainCalls))
	copy(out, w.trainCalls)
	return out
}

// PredictCalls returns a snapshot of every PredictRequest received so far.
func (w *FakeWorker) PredictCalls() []*pb.PredictRequest {
	w.mu.Lock()
	defer w.mu.Unlock()
	out := make([]*pb.PredictRequest, len(w.predictCalls))
	copy(out, w.predictCalls)
	return out
}

func (w *FakeWorker) Health(_ context.Context, _ *pb.HealthRequest) (*pb.HealthResponse, error) {
	w.mu.Lock()
	defer w.mu.Unlock()
	return &pb.HealthResponse{Healthy: w.healthy}, nil
}

func (w *FakeWorker) Train(ctx context.Context, req *pb.TrainRequest) (*pb.TrainResponse, error) {
	w.mu.Lock()
	w.trainCalls = append(w.trainCalls, req)
	fn := w.TrainFunc
	w.mu.Unlock()

	if fn != nil {
		return fn(ctx, req)
	}
	return &pb.TrainResponse{Success: true, Message: "ok"}, nil
}

func (w *FakeWorker) Predict(ctx context.Context, req *pb.PredictRequest) (*pb.PredictResponse, error) {
	w.mu.Lock()
	w.predictCalls = append(w.predictCalls, req)
	fn := w.PredictFunc
	w.mu.Unlock()

	if fn != nil {
		return fn(ctx, req)
	}
	return PredictFuncReturning(ClassVote("0"))(ctx, req)
}
