package orchestrator

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"os"
	"regexp"
	"strconv"

	"github.com/aws/aws-sdk-go-v2/aws"
	awsconfig "github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/credentials"
	"github.com/aws/aws-sdk-go-v2/service/s3"
	"github.com/aws/aws-sdk-go-v2/service/s3/types"
	"github.com/aws/smithy-go"

	"github.com/bytelisa/distributed-random-forest/internal/config"
)

// Centralizing everything needed by the master to read/write on S3: building the
// client, listing keys, and reading/writing small JSON objects.
// Used by:
//  - persisting train request metadata (pool.go)
//  - scanning for incomplete trainings (reconcile.go)
//  - answering a status query (status.go)

// S3Store wraps an S3 client scoped to a single bucket.
type S3Store struct {
	client *s3.Client
	bucket string
}

// resolveRegion picks the AWS region used to sign requests.
func resolveRegion() string {
	if region := os.Getenv("AWS_REGION"); region != "" {
		return region
	}
	if region := os.Getenv("AWS_DEFAULT_REGION"); region != "" {
		return region
	}
	return "us-east-1"
}

// NewS3Store builds an S3 client from the given storage config.
func NewS3Store(ctx context.Context, storageCfg *config.StorageConfig) (*S3Store, error) {
	var optFns []func(*awsconfig.LoadOptions) error
	if storageCfg.AccessKey != "" && storageCfg.SecretKey != "" {
		optFns = append(optFns, awsconfig.WithCredentialsProvider(
			credentials.NewStaticCredentialsProvider(storageCfg.AccessKey, storageCfg.SecretKey, ""),
		))
	}

	optFns = append(optFns, awsconfig.WithRegion(resolveRegion()))

	cfg, err := awsconfig.LoadDefaultConfig(ctx, optFns...)
	if err != nil {
		return nil, err
	}

	client := s3.NewFromConfig(cfg, func(o *s3.Options) {
		if storageCfg.Endpoint != "" {
			o.BaseEndpoint = aws.String(storageCfg.Endpoint)
		}
		o.UsePathStyle = true // needed for MinIO; harmless for real S3
	})

	return &S3Store{client: client, bucket: storageCfg.Bucket}, nil
}

// ListKeys returns every object key under the given prefix.
func (s *S3Store) ListKeys(ctx context.Context, prefix string) ([]string, error) {
	var keys []string
	paginator := s3.NewListObjectsV2Paginator(s.client, &s3.ListObjectsV2Input{
		Bucket: aws.String(s.bucket),
		Prefix: aws.String(prefix),
	})
	for paginator.HasMorePages() {
		page, err := paginator.NextPage(ctx)
		if err != nil {
			return nil, err
		}
		for _, obj := range page.Contents {
			keys = append(keys, aws.ToString(obj.Key))
		}
	}
	return keys, nil
}

// ListCommonPrefixes returns the "folders" directly under prefix (using
// delimiter "/"), without descending into them - e.g.
// ListCommonPrefixes(ctx, "models/") returns "models/<id>/" for every model_id
func (s *S3Store) ListCommonPrefixes(ctx context.Context, prefix string) ([]string, error) {
	var prefixes []string
	paginator := s3.NewListObjectsV2Paginator(s.client, &s3.ListObjectsV2Input{
		Bucket:    aws.String(s.bucket),
		Prefix:    aws.String(prefix),
		Delimiter: aws.String("/"),
	})
	for paginator.HasMorePages() {
		page, err := paginator.NextPage(ctx)
		if err != nil {
			return nil, err
		}
		for _, cp := range page.CommonPrefixes {
			prefixes = append(prefixes, aws.ToString(cp.Prefix))
		}
	}
	return prefixes, nil
}

// GetJSON reads and unmarshals a JSON object into the variable out.
// Returns found=false (no error) if the key doesn't exist.
func (s *S3Store) GetJSON(ctx context.Context, key string, out interface{}) (found bool, err error) {
	body, found, err := s.GetBytes(ctx, key)
	if err != nil || !found {
		return found, err
	}
	if err := json.Unmarshal(body, out); err != nil {
		return false, err
	}
	return true, nil
}

// GetBytes reads the raw bytes of an object. Returns found=false (no error)
// if the key doesn't exist.
func (s *S3Store) GetBytes(ctx context.Context, key string) (body []byte, found bool, err error) {
	resp, err := s.client.GetObject(ctx, &s3.GetObjectInput{
		Bucket: aws.String(s.bucket),
		Key:    aws.String(key),
	})
	if err != nil {
		var noSuchKey *types.NoSuchKey
		if errors.As(err, &noSuchKey) {
			return nil, false, nil
		}
		// Some S3-compatible backends (MinIO) surface a generic API error
		// instead of the typed NoSuchKey.
		var apiErr smithy.APIError
		if errors.As(err, &apiErr) && (apiErr.ErrorCode() == "NoSuchKey" || apiErr.ErrorCode() == "NotFound") {
			return nil, false, nil
		}
		return nil, false, err
	}
	defer resp.Body.Close()

	body, err = io.ReadAll(resp.Body)
	if err != nil {
		return nil, false, err
	}
	return body, true, nil
}

// PutJSON marshals data to JSON and uploads it to variable key.
func (s *S3Store) PutJSON(ctx context.Context, key string, data interface{}) error {
	body, err := json.Marshal(data)
	if err != nil {
		return err
	}
	return s.PutBytes(ctx, key, body)
}

// PutBytes uploads raw bytes to the given key.
func (s *S3Store) PutBytes(ctx context.Context, key string, body []byte) error {
	_, err := s.client.PutObject(ctx, &s3.PutObjectInput{
		Bucket: aws.String(s.bucket),
		Key:    aws.String(key),
		Body:   bytes.NewReader(body),
	})
	return err
}

// TrainRequestMetadata is the schema of models/{model_id}/train_request.json:
type TrainRequestMetadata struct {
	DatasetURL      string            `json:"dataset_url"`
	TaskType        int32             `json:"task_type"`
	TargetColumn    string            `json:"target_column"`
	NEstimators     int32             `json:"n_estimators"`
	TotalPartitions int32             `json:"total_partitions"`
	Hyperparameters map[string]string `json:"hyperparameters,omitempty"`
}

// partIndexRE matches the deterministic model part filenames written by
// the worker: forest_part_{index}.joblib.
var partIndexRE = regexp.MustCompile(`forest_part_(\d+)\.joblib$`)

// parsePartitionIndex extracts the partition index from a model part S3
// key, if it matches the expected naming scheme.
func parsePartitionIndex(key string) (int32, bool) {
	match := partIndexRE.FindStringSubmatch(key)
	if match == nil {
		return 0, false
	}
	idx, err := strconv.Atoi(match[1])
	if err != nil {
		return 0, false
	}
	return int32(idx), true
}
