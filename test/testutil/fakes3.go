package testutil

import (
	"encoding/xml"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"sort"
	"strings"
	"sync"
	"testing"
)

// FakeS3 is a minimal in-memory S3-compatible HTTP server: just enough of
// PutObject/GetObject/ListObjectsV2 (path-style addressing, as
// internal/orchestrator.S3Store configures) to exercise the master's
// storage code without a real MinIO/S3 endpoint or network access.
type FakeS3 struct {
	mu      sync.Mutex
	objects map[string][]byte

	server *httptest.Server
}

// NewFakeS3 starts the fake server and stops it via t.Cleanup.
func NewFakeS3(t *testing.T) *FakeS3 {
	t.Helper()

	f := &FakeS3{objects: make(map[string][]byte)}
	f.server = httptest.NewServer(http.HandlerFunc(f.handle))
	t.Cleanup(f.server.Close)
	return f
}

// Endpoint is the base URL to set as config.StorageConfig.Endpoint.
func (f *FakeS3) Endpoint() string {
	return f.server.URL
}

// Put seeds an object directly (bypassing HTTP), for test setup.
func (f *FakeS3) Put(key string, body []byte) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.objects[key] = body
}

// Has reports whether an object exists, for assertions.
func (f *FakeS3) Has(key string) bool {
	f.mu.Lock()
	defer f.mu.Unlock()
	_, ok := f.objects[key]
	return ok
}

// Keys returns every stored key with the given prefix, sorted.
func (f *FakeS3) Keys(prefix string) []string {
	f.mu.Lock()
	defer f.mu.Unlock()

	var keys []string
	for k := range f.objects {
		if strings.HasPrefix(k, prefix) {
			keys = append(keys, k)
		}
	}
	sort.Strings(keys)
	return keys
}

func (f *FakeS3) handle(w http.ResponseWriter, r *http.Request) {
	// Path-style addressing: /{bucket}/{key...}
	path := strings.TrimPrefix(r.URL.Path, "/")
	parts := strings.SplitN(path, "/", 2)
	bucket := parts[0]

	switch r.Method {
	case http.MethodPut:
		if len(parts) < 2 || parts[1] == "" {
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		f.handlePut(w, r, parts[1])

	case http.MethodGet:
		if len(parts) < 2 || parts[1] == "" {
			f.handleList(w, r, bucket)
			return
		}
		f.handleGet(w, parts[1])

	default:
		w.WriteHeader(http.StatusMethodNotAllowed)
	}
}

func (f *FakeS3) handlePut(w http.ResponseWriter, r *http.Request, key string) {
	body, err := io.ReadAll(r.Body)
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		return
	}

	f.mu.Lock()
	f.objects[key] = body
	f.mu.Unlock()

	w.Header().Set("ETag", `"fake-etag"`)
	w.WriteHeader(http.StatusOK)
}

func (f *FakeS3) handleGet(w http.ResponseWriter, key string) {
	f.mu.Lock()
	body, ok := f.objects[key]
	f.mu.Unlock()

	if !ok {
		f.writeNoSuchKey(w, key)
		return
	}

	w.Header().Set("ETag", `"fake-etag"`)
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write(body)
}

func (f *FakeS3) writeNoSuchKey(w http.ResponseWriter, key string) {
	w.Header().Set("Content-Type", "application/xml")
	w.WriteHeader(http.StatusNotFound)
	_, _ = fmt.Fprintf(w, `<?xml version="1.0" encoding="UTF-8"?>`+
		`<Error><Code>NoSuchKey</Code><Message>The specified key does not exist.</Message><Key>%s</Key></Error>`,
		xmlEscape(key))
}

type listBucketResult struct {
	XMLName        xml.Name       `xml:"ListBucketResult"`
	Name           string         `xml:"Name"`
	Prefix         string         `xml:"Prefix"`
	Delimiter      string         `xml:"Delimiter,omitempty"`
	KeyCount       int            `xml:"KeyCount"`
	MaxKeys        int            `xml:"MaxKeys"`
	IsTruncated    bool           `xml:"IsTruncated"`
	Contents       []listContent  `xml:"Contents"`
	CommonPrefixes []commonPrefix `xml:"CommonPrefixes"`
}

type listContent struct {
	Key          string `xml:"Key"`
	LastModified string `xml:"LastModified"`
	ETag         string `xml:"ETag"`
	Size         int64  `xml:"Size"`
	StorageClass string `xml:"StorageClass"`
}

type commonPrefix struct {
	Prefix string `xml:"Prefix"`
}

// handleList implements just enough of ListObjectsV2 (prefix + delimiter,
// no pagination - tests never store enough keys to need it) to back
// S3Store.ListKeys and S3Store.ListCommonPrefixes.
func (f *FakeS3) handleList(w http.ResponseWriter, r *http.Request, bucket string) {
	q := r.URL.Query()
	prefix := q.Get("prefix")
	delimiter := q.Get("delimiter")

	f.mu.Lock()
	type entry struct {
		key string
		len int
	}
	var entries []entry
	for k, v := range f.objects {
		if strings.HasPrefix(k, prefix) {
			entries = append(entries, entry{key: k, len: len(v)})
		}
	}
	f.mu.Unlock()

	sort.Slice(entries, func(i, j int) bool { return entries[i].key < entries[j].key })

	result := listBucketResult{
		Name:      bucket,
		Prefix:    prefix,
		Delimiter: delimiter,
		MaxKeys:   1000,
	}

	seenPrefixes := make(map[string]bool)
	for _, e := range entries {
		if delimiter != "" {
			rest := strings.TrimPrefix(e.key, prefix)
			if idx := strings.Index(rest, delimiter); idx >= 0 {
				cp := prefix + rest[:idx+len(delimiter)]
				if !seenPrefixes[cp] {
					seenPrefixes[cp] = true
					result.CommonPrefixes = append(result.CommonPrefixes, commonPrefix{Prefix: cp})
				}
				continue
			}
		}
		result.Contents = append(result.Contents, listContent{
			Key:          e.key,
			LastModified: "2024-01-01T00:00:00.000Z",
			ETag:         `"fake-etag"`,
			Size:         int64(e.len),
			StorageClass: "STANDARD",
		})
	}
	result.KeyCount = len(result.Contents) + len(result.CommonPrefixes)

	w.Header().Set("Content-Type", "application/xml")
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write([]byte(xml.Header))
	_ = xml.NewEncoder(w).Encode(result)
}

func xmlEscape(s string) string {
	var b strings.Builder
	_ = xml.EscapeText(&b, []byte(s))
	return b.String()
}
