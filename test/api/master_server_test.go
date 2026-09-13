// Black-box HTTP tests for internal/api.Server, driven directly through
// Server.Router() (no real port bound) against fake gRPC workers and a fake
// in-memory S3. See test_suite_design.md, sezione E (E1-E5).
package api_test

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	pb "github.com/bytelisa/distributed-random-forest/api/proto/worker/v1"
	"github.com/bytelisa/distributed-random-forest/internal/api"
	"github.com/bytelisa/distributed-random-forest/internal/config"
	"github.com/bytelisa/distributed-random-forest/test/testutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

func newTestServer(t *testing.T) (*api.Server, *testutil.FakeWorker, *testutil.FakeS3) {
	t.Helper()

	worker := testutil.NewFakeWorker(t)
	fakeS3 := testutil.NewFakeS3(t)
	storageCfg := testutil.NewTestStorageConfig(fakeS3.Endpoint(), "bucket")

	cfg := &config.Config{
		Workers: config.WorkerConfig{Addresses: []string{worker.Address}},
		Storage: storageCfg,
		System: config.SystemConfig{
			TimeoutTraining:    5,
			TimeoutPrediction:  5,
			TimeoutHealthCheck: 2,
		},
	}

	srv, err := api.NewServer(cfg)
	require.NoError(t, err)
	srv.WorkerPool().PartitionDataset = testutil.FakePartitionDataset

	return srv, worker, fakeS3
}

func doJSON(t *testing.T, srv *api.Server, method, path, body string) *httptest.ResponseRecorder {
	t.Helper()
	req := httptest.NewRequest(method, path, strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	srv.Router().ServeHTTP(rec, req)
	return rec
}

// ---------------------------------------------------------------------
// E1 — POST /train con task_type non valido o campi obbligatori mancanti
// ---------------------------------------------------------------------

func TestE1_TrainValidation(t *testing.T) {
	srv, worker, _ := newTestServer(t)

	cases := []struct {
		name string
		body string
	}{
		{"missing dataset_url", `{"task_type":"classification","target_column":"y"}`},
		{"missing target_column", `{"task_type":"classification","dataset_url":"data/iris.csv"}`},
		{"invalid task_type", `{"task_type":"bogus","dataset_url":"data/iris.csv","target_column":"y"}`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			rec := doJSON(t, srv, http.MethodPost, "/train", tc.body)
			assert.Equal(t, http.StatusBadRequest, rec.Code)
		})
	}

	// None of the invalid requests should have reached the worker.
	assert.Empty(t, worker.TrainCalls())
}

// ---------------------------------------------------------------------
// E2 — GET /models/{id} per un id sconosciuto
// ---------------------------------------------------------------------

func TestE2_ModelStatusUnknownID(t *testing.T) {
	srv, _, _ := newTestServer(t)

	req := httptest.NewRequest(http.MethodGet, "/models/does-not-exist", nil)
	rec := httptest.NewRecorder()
	srv.Router().ServeHTTP(rec, req)

	assert.Equal(t, http.StatusNotFound, rec.Code)

	var body map[string]any
	require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &body))
	assert.Equal(t, "does-not-exist", body["model_id"])
}

// ---------------------------------------------------------------------
// E3 — POST /train risponde subito e il training continua in background
// ---------------------------------------------------------------------

func TestE3_TrainRespondsImmediatelyAndTrainsInBackground(t *testing.T) {
	srv, worker, fakeS3 := newTestServer(t)
	storageCfg := testutil.NewTestStorageConfig(fakeS3.Endpoint(), "bucket")
	uploadPart := testutil.TrainFuncUploadingPart(&storageCfg)

	block := make(chan struct{})
	started := make(chan struct{})
	worker.TrainFunc = func(ctx context.Context, req *pb.TrainRequest) (*pb.TrainResponse, error) {
		close(started)
		<-block // held open until the test explicitly unblocks it
		return uploadPart(ctx, req)
	}

	body := `{"task_type":"classification","dataset_url":"data/iris.csv","target_column":"species"}`
	req := httptest.NewRequest(http.MethodPost, "/train", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()

	done := make(chan struct{})
	go func() {
		srv.Router().ServeHTTP(rec, req)
		close(done)
	}()

	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("handler did not return promptly - it must not wait for training to finish")
	}

	require.Equal(t, http.StatusAccepted, rec.Code)

	var resp api.TrainResponse
	require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &resp))
	assert.NotEmpty(t, resp.ModelID)
	assert.Equal(t, "training", resp.Status)

	select {
	case <-started:
	case <-time.After(2 * time.Second):
		t.Fatal("background training never started")
	}

	// GET /models/{id} sees "training" while the worker is still blocked -
	// the request context that produced the 202 is long gone by now, which
	// proves the background goroutine runs on its own context, not the
	// (already-cancelled) request's one.
	statusRec := doJSON(t, srv, http.MethodGet, "/models/"+resp.ModelID, "")
	require.Equal(t, http.StatusOK, statusRec.Code)
	var statusResp api.TrainResponse
	require.NoError(t, json.Unmarshal(statusRec.Body.Bytes(), &statusResp))
	assert.Equal(t, "training", statusResp.Status)

	close(block)

	// The same model_id from the immediate response eventually becomes
	// observable as "ready" through GET /models/{id}.
	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		rec := doJSON(t, srv, http.MethodGet, "/models/"+resp.ModelID, "")
		if rec.Code == http.StatusOK {
			var got api.TrainResponse
			require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &got))
			if got.Status != "training" {
				return
			}
		}
		time.Sleep(20 * time.Millisecond)
	}
	t.Fatal("model never left the \"training\" status")
}

// ---------------------------------------------------------------------
// E4 — POST /predict/{id} con features mancanti o task_type non valido
// ---------------------------------------------------------------------

func TestE4_PredictValidation(t *testing.T) {
	srv, worker, _ := newTestServer(t)

	cases := []struct {
		name string
		body string
	}{
		{"missing features", `{"task_type":"classification"}`},
		{"invalid task_type", `{"task_type":"bogus","features":[1,2,3]}`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			rec := doJSON(t, srv, http.MethodPost, "/predict/some-model", tc.body)
			assert.Equal(t, http.StatusBadRequest, rec.Code)
		})
	}

	assert.Empty(t, worker.PredictCalls())
}

// ---------------------------------------------------------------------
// E5 — POST /predict/{id} su modello non pronto o inesistente
// ---------------------------------------------------------------------

func TestE5_PredictOnNonexistentModel(t *testing.T) {
	srv, worker, _ := newTestServer(t)

	// Same failure a real worker produces today when it finds no model
	// parts for this model_id (see worker_service.py:140-142 /
	// worker.proto:66-69) - PredictDistributed treats it like any other
	// failed worker.
	worker.PredictFunc = func(_ context.Context, _ *pb.PredictRequest) (*pb.PredictResponse, error) {
		return nil, status.Error(codes.Internal, `Protocol message PredictResponse has no "prediction" field.`)
	}

	body := `{"task_type":"classification","features":[1,2,3]}`
	rec := doJSON(t, srv, http.MethodPost, "/predict/does-not-exist", body)

	require.Equal(t, http.StatusInternalServerError, rec.Code)

	var got map[string]any
	require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &got))
	assert.Contains(t, got["error"], "Distributed inference failed")
	assert.Contains(t, got["error"], "no workers returned valid results")
}
