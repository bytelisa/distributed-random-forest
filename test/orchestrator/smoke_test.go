package orchestrator_test

import (
	"context"
	"testing"

	"github.com/bytelisa/distributed-random-forest/internal/orchestrator"
	"github.com/bytelisa/distributed-random-forest/test/testutil"
)

// TestSmoke_S3Store_AgainstFakeS3 is not one of the designed test cases; it
// exists to validate the FakeS3 test double itself (XML responses the AWS
// SDK v2 client can actually parse) before relying on it throughout the
// rest of this package.
func TestSmoke_S3Store_AgainstFakeS3(t *testing.T) {
	fakeS3 := testutil.NewFakeS3(t)
	storageCfg := testutil.NewTestStorageConfig(fakeS3.Endpoint(), "test-bucket")

	ctx := context.Background()
	store, err := orchestrator.NewS3Store(ctx, &storageCfg)
	if err != nil {
		t.Fatalf("NewS3Store: %v", err)
	}

	// GetJSON on a missing key: found=false, no error.
	var out map[string]any
	found, err := store.GetJSON(ctx, "models/m1/train_request.json", &out)
	if err != nil {
		t.Fatalf("GetJSON on missing key: %v", err)
	}
	if found {
		t.Fatalf("expected found=false for missing key")
	}

	// PutJSON then GetJSON round-trip.
	type meta struct {
		TotalPartitions int32 `json:"total_partitions"`
	}
	if err := store.PutJSON(ctx, "models/m1/train_request.json", meta{TotalPartitions: 3}); err != nil {
		t.Fatalf("PutJSON: %v", err)
	}
	var got meta
	found, err = store.GetJSON(ctx, "models/m1/train_request.json", &got)
	if err != nil || !found {
		t.Fatalf("GetJSON after PutJSON: found=%v err=%v", found, err)
	}
	if got.TotalPartitions != 3 {
		t.Fatalf("expected TotalPartitions=3, got %d", got.TotalPartitions)
	}

	// ListKeys with a prefix.
	if err := store.PutBytes(ctx, "models/m1/model_parts/forest_part_0.joblib", []byte("x")); err != nil {
		t.Fatalf("PutBytes: %v", err)
	}
	if err := store.PutBytes(ctx, "models/m1/model_parts/forest_part_1.joblib", []byte("y")); err != nil {
		t.Fatalf("PutBytes: %v", err)
	}
	keys, err := store.ListKeys(ctx, "models/m1/model_parts/")
	if err != nil {
		t.Fatalf("ListKeys: %v", err)
	}
	if len(keys) != 2 {
		t.Fatalf("expected 2 keys, got %d: %v", len(keys), keys)
	}

	// ListCommonPrefixes with a delimiter.
	prefixes, err := store.ListCommonPrefixes(ctx, "models/")
	if err != nil {
		t.Fatalf("ListCommonPrefixes: %v", err)
	}
	if len(prefixes) != 1 || prefixes[0] != "models/m1/" {
		t.Fatalf("expected [\"models/m1/\"], got %v", prefixes)
	}
}
