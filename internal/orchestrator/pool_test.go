package orchestrator

// White-box unit tests (same package: aggregateRegression/aggregateClassification
// and splitIndices are unexported) for the final-aggregation strategies used
// by PredictDistributed and for the tree assignment. See test_suite_design.md,
// sezione A, A1/A2.

import (
	"testing"

	pb "github.com/bytelisa/distributed-random-forest/api/proto/worker/v1"
	"github.com/stretchr/testify/assert"
)

// A1 — aggregateRegression
func TestAggregateRegression(t *testing.T) {
	t.Run("mixed numeric values returns the mean", func(t *testing.T) {
		got := aggregateRegression([]float64{1, 2, 3})
		assert.Equal(t, "2.000000", got)
	})

	t.Run("empty list returns zero instead of dividing by zero", func(t *testing.T) {
		got := aggregateRegression(nil)
		assert.Equal(t, "0", got)
	})
}

func vote(classes []string, probs []float64) *pb.TreePrediction {
	return &pb.TreePrediction{Classes: classes, Probabilities: probs}
}

// A2 — aggregateClassification (soft voting)
func TestAggregateClassification(t *testing.T) {
	t.Run("class with the highest summed probability wins", func(t *testing.T) {
		got := aggregateClassification([]*pb.TreePrediction{
			vote([]string{"cat", "dog"}, []float64{0.6, 0.4}),
			vote([]string{"cat", "dog"}, []float64{0.2, 0.8}),
			vote([]string{"cat", "dog"}, []float64{0.9, 0.1}),
		})
		assert.Equal(t, "cat", got, "cat: 1.7 vs dog: 1.3")
	})

	t.Run("soft voting differs from majority vote of hard labels", func(t *testing.T) {
		// Two trees lean slightly to "dog", one is certain about "cat":
		// hard voting would say "dog", summed probabilities say "cat".
		got := aggregateClassification([]*pb.TreePrediction{
			vote([]string{"cat", "dog"}, []float64{0.45, 0.55}),
			vote([]string{"cat", "dog"}, []float64{0.45, 0.55}),
			vote([]string{"cat", "dog"}, []float64{1.0, 0.0}),
		})
		assert.Equal(t, "cat", got)
	})

	t.Run("trees that never saw a class contribute zero for it", func(t *testing.T) {
		// A tree trained on a bootstrap sample without "bird" only reports
		// the classes it knows; aggregation must align by label, not position.
		got := aggregateClassification([]*pb.TreePrediction{
			vote([]string{"bird", "cat"}, []float64{0.7, 0.3}),
			vote([]string{"cat"}, []float64{1.0}),
			vote([]string{"bird", "cat", "dog"}, []float64{0.5, 0.3, 0.2}),
		})
		assert.Equal(t, "cat", got, "cat: 1.6 vs bird: 1.2 vs dog: 0.2")
	})

	t.Run("single tree is trivially the answer", func(t *testing.T) {
		got := aggregateClassification([]*pb.TreePrediction{vote([]string{"only"}, []float64{1})})
		assert.Equal(t, "only", got)
	})

	t.Run("tie goes to the first class in sorted order", func(t *testing.T) {
		got := aggregateClassification([]*pb.TreePrediction{vote([]string{"dog", "cat"}, []float64{0.5, 0.5})})
		assert.Equal(t, "cat", got)
	})

	t.Run("no predictions returns an empty label", func(t *testing.T) {
		assert.Equal(t, "", aggregateClassification(nil))
	})
}

// splitIndices — tree assignment among workers
func TestSplitIndices(t *testing.T) {
	t.Run("even split", func(t *testing.T) {
		got := splitIndices(rangeIndices(6), 3)
		assert.Equal(t, [][]int32{{0, 1}, {2, 3}, {4, 5}}, got)
	})

	t.Run("remainder goes to the first chunks", func(t *testing.T) {
		got := splitIndices(rangeIndices(10), 3)
		assert.Equal(t, [][]int32{{0, 1, 2, 3}, {4, 5, 6}, {7, 8, 9}}, got)
	})

	t.Run("more workers than trees leaves empty chunks", func(t *testing.T) {
		got := splitIndices(rangeIndices(2), 3)
		assert.Len(t, got, 3)
		assert.Equal(t, []int32{0}, got[0])
		assert.Equal(t, []int32{1}, got[1])
		assert.Empty(t, got[2])
	})

	t.Run("non-contiguous input (a retry) is split as given", func(t *testing.T) {
		got := splitIndices([]int32{1, 4, 9}, 2)
		assert.Equal(t, [][]int32{{1, 4}, {9}}, got)
	})
}
