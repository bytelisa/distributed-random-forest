package orchestrator

// White-box unit tests (same package: parseTreeIndex/resolveRegion are
// unexported) for the two pure helpers in s3store.go.
// See test_suite_design.md, sezione A, A4/A6.

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

// A4 — parseTreeIndex
func TestParseTreeIndex(t *testing.T) {
	cases := []struct {
		name    string
		key     string
		wantIdx int32
		wantOk  bool
	}{
		{
			name:    "valid single-digit index",
			key:     "models/abc/model_parts/tree_0.joblib",
			wantIdx: 0,
			wantOk:  true,
		},
		{
			name:    "valid multi-digit index",
			key:     "models/abc/model_parts/tree_12.joblib",
			wantIdx: 12,
			wantOk:  true,
		},
		{
			name:   "per-worker naming does not match",
			key:    "models/abc/model_parts/forest_part_0.joblib",
			wantOk: false,
		},
		{
			name:   "unrelated key does not match",
			key:    "models/abc/train_request.json",
			wantOk: false,
		},
		{
			name:   "trailing suffix after .joblib does not match (regex is anchored at $)",
			key:    "models/abc/model_parts/tree_0.joblib.bak",
			wantOk: false,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			idx, ok := parseTreeIndex(tc.key)
			assert.Equal(t, tc.wantOk, ok)
			if tc.wantOk {
				assert.Equal(t, tc.wantIdx, idx)
			}
		})
	}
}

// A6 — resolveRegion
func TestResolveRegion(t *testing.T) {
	t.Run("defaults to us-east-1 when neither env var is set", func(t *testing.T) {
		t.Setenv("AWS_REGION", "")
		t.Setenv("AWS_DEFAULT_REGION", "")
		assert.Equal(t, "us-east-1", resolveRegion())
	})

	t.Run("uses AWS_REGION when set", func(t *testing.T) {
		t.Setenv("AWS_REGION", "eu-west-1")
		t.Setenv("AWS_DEFAULT_REGION", "")
		assert.Equal(t, "eu-west-1", resolveRegion())
	})

	t.Run("falls back to AWS_DEFAULT_REGION when AWS_REGION is unset", func(t *testing.T) {
		t.Setenv("AWS_REGION", "")
		t.Setenv("AWS_DEFAULT_REGION", "ap-southeast-2")
		assert.Equal(t, "ap-southeast-2", resolveRegion())
	})

	t.Run("AWS_REGION takes priority over AWS_DEFAULT_REGION", func(t *testing.T) {
		t.Setenv("AWS_REGION", "eu-west-1")
		t.Setenv("AWS_DEFAULT_REGION", "ap-southeast-2")
		assert.Equal(t, "eu-west-1", resolveRegion())
	})
}
