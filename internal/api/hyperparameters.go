package api

import (
	"fmt"
	"math"
	"strconv"
)

// This file validates the generic `hyperparameters` map from a training
// request as soon as it's received
//
// Converts every value to a string and does some simple validation

// allowedHyperparameters lists every RandomForest hyperparameter accepted
// through the generic `hyperparameters` field, beyond n_estimators
var allowedHyperparameters = map[string]bool{
	"max_depth":         true,
	"max_features":      true,
	"min_samples_split": true,
	"min_samples_leaf":  true,
	"bootstrap":         true,
	"max_samples":       true,
	"criterion":         true,
	"max_leaf_nodes":    true,
	"class_weight":      true,
	"oob_score":         true,
}

// validateHyperparameters checks that every key is supported and has a
// plausible value for the given task_type, and converts everything to
// strings for the gRPC connection.
func validateHyperparameters(raw map[string]interface{}, taskType string) (map[string]string, error) {
	result := make(map[string]string, len(raw))

	for key, value := range raw {
		if !allowedHyperparameters[key] {
			return nil, fmt.Errorf("unsupported hyperparameter %q", key)
		}

		strValue, err := validateHyperparameter(key, value, taskType)
		if err != nil {
			return nil, fmt.Errorf("invalid hyperparameter %q: %w", key, err)
		}
		result[key] = strValue
	}

	// scikit-learn requires bootstrap=True for oob_score
	if oob, hasOOB := result["oob_score"]; hasOOB && oob == "true" && result["bootstrap"] != "true" {
		return nil, fmt.Errorf("oob_score=true requires bootstrap=true")
	}

	return result, nil
}

func validateHyperparameter(key string, value interface{}, taskType string) (string, error) {
	switch key {
	case "max_depth", "max_leaf_nodes":
		return validatePositiveIntOrNull(value)
	case "min_samples_split":
		return validateIntOrFraction(value, 2)
	case "min_samples_leaf":
		return validateIntOrFraction(value, 1)
	case "max_features":
		return validateMaxFeatures(value)
	case "max_samples":
		return validatePositiveIntOrFractionOrNull(value)
	case "bootstrap", "oob_score":
		return validateBool(value)
	case "criterion":
		return validateCriterion(value, taskType)
	case "class_weight":
		return validateClassWeight(value, taskType)
	default:
		// Unreachable: callers already check allowedHyperparameters first.
		return "", fmt.Errorf("unsupported hyperparameter")
	}
}

func isNull(value interface{}) bool {
	return value == nil
}

func validateBool(value interface{}) (string, error) {
	b, ok := value.(bool)
	if !ok {
		return "", fmt.Errorf("expected a boolean")
	}
	return strconv.FormatBool(b), nil
}

// validatePositiveIntOrNull accepts a positive integer or null
func validatePositiveIntOrNull(value interface{}) (string, error) {
	if isNull(value) {
		return "None", nil
	}
	n, ok := value.(float64) // JSON numbers decode into float64
	if !ok || n != math.Trunc(n) || n < 1 {
		return "", fmt.Errorf("expected a positive integer or null")
	}
	return strconv.Itoa(int(n)), nil
}

// validateIntOrFraction accepts either an integer >= min, or a float in (0, 1]
func validateIntOrFraction(value interface{}, min int) (string, error) {
	n, ok := value.(float64)
	if !ok {
		return "", fmt.Errorf("expected a number")
	}
	if n == math.Trunc(n) && n >= float64(min) {
		return strconv.Itoa(int(n)), nil
	}
	if n > 0 && n <= 1 {
		return strconv.FormatFloat(n, 'f', -1, 64), nil
	}
	return "", fmt.Errorf("expected an integer >= %d or a fraction in (0, 1]", min)
}

func validatePositiveIntOrFractionOrNull(value interface{}) (string, error) {
	if isNull(value) {
		return "None", nil
	}
	return validateIntOrFraction(value, 1)
}

// validateMaxFeatures accepts "sqrt", "log2", a positive integer, a fraction in (0, 1], or null.
func validateMaxFeatures(value interface{}) (string, error) {
	if isNull(value) {
		return "None", nil
	}
	if s, ok := value.(string); ok {
		if s == "sqrt" || s == "log2" {
			return s, nil
		}
		return "", fmt.Errorf(`expected "sqrt", "log2", a number, or null`)
	}
	if n, ok := value.(float64); ok {
		if n == math.Trunc(n) && n >= 1 {
			return strconv.Itoa(int(n)), nil
		}
		if n > 0 && n <= 1 {
			return strconv.FormatFloat(n, 'f', -1, 64), nil
		}
	}
	return "", fmt.Errorf(`expected "sqrt", "log2", a number, or null`)
}

func validateCriterion(value interface{}, taskType string) (string, error) {
	s, ok := value.(string)
	if !ok {
		return "", fmt.Errorf("expected a string")
	}

	var allowed map[string]bool
	if taskType == "classification" {
		allowed = map[string]bool{"gini": true, "entropy": true, "log_loss": true}
	} else {
		allowed = map[string]bool{"squared_error": true, "absolute_error": true, "friedman_mse": true, "poisson": true}
	}

	if !allowed[s] {
		return "", fmt.Errorf("invalid criterion %q for task_type %q", s, taskType)
	}
	return s, nil
}

// validateClassWeight only makes sense for classification: "balanced", "balanced_subsample", or null.
func validateClassWeight(value interface{}, taskType string) (string, error) {
	if taskType != "classification" {
		return "", fmt.Errorf("class_weight is only valid for classification")
	}
	if isNull(value) {
		return "None", nil
	}
	s, ok := value.(string)
	if !ok || (s != "balanced" && s != "balanced_subsample") {
		return "", fmt.Errorf(`expected "balanced", "balanced_subsample", or null`)
	}
	return s, nil
}
