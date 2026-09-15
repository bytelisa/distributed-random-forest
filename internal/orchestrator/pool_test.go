package orchestrator

// White-box unit tests (same package: aggregateRegression/aggregateClassification
// are unexported) for the two final-aggregation strategies used by
// PredictDistributed. See test_suite_design.md, sezione A, A1/A2.

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

// A1 — aggregateRegression
func TestAggregateRegression(t *testing.T) {
	t.Run("mixed numeric values returns the mean", func(t *testing.T) {
		got := aggregateRegression([]string{"1", "2", "3"})
		assert.Equal(t, "2.000000", got)
	})

	t.Run("unparsable values are skipped, not fatal", func(t *testing.T) {
		got := aggregateRegression([]string{"1", "not-a-number", "3"})
		assert.Equal(t, "2.000000", got, "mean should be computed only over the valid values")
	})

	t.Run("all values unparsable behaves like an empty list", func(t *testing.T) {
		// Note: "nan"/"inf" are NOT good fixtures here - strconv.ParseFloat
		// accepts them as valid floats (producing a NaN/Inf mean), so they
		// don't exercise this branch at all.
		got := aggregateRegression([]string{"foo", "bar"})
		assert.Equal(t, "0", got)
	})

	t.Run("empty list returns zero instead of dividing by zero", func(t *testing.T) {
		got := aggregateRegression(nil)
		assert.Equal(t, "0", got)
	})
}

// A2 — aggregateClassification
func TestAggregateClassification(t *testing.T) {
	t.Run("clear majority wins", func(t *testing.T) {
		got := aggregateClassification([]string{"cat", "dog", "cat", "cat", "dog"})
		assert.Equal(t, "cat", got)
	})

	t.Run("single value is trivially the majority", func(t *testing.T) {
		got := aggregateClassification([]string{"only"})
		assert.Equal(t, "only", got)
	})

	t.Run("tie: result is one of the tied candidates, not a fixed value", func(t *testing.T) {
		// Go map iteration order is randomized, so which of the tied
		// candidates wins is not deterministic across runs - the contract
		// this test pins down is only "it's one of them", not a specific one.
		got := aggregateClassification([]string{"cat", "dog"})
		assert.Contains(t, []string{"cat", "dog"}, got)
	})
}
