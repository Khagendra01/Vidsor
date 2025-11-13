# GPT-4o-mini Batch Processing Guide

## Yes, you can do batch processing with GPT-4o-mini!

There are **two main approaches** for batch processing with GPT-4o-mini:

---

## Option 1: Concurrent Requests (Recommended for Real-Time Processing)

**What it is:** Process multiple requests in parallel using `ThreadPoolExecutor`.

**Pros:**
- ✅ Real-time processing (get results as they complete)
- ✅ Faster than sequential processing
- ✅ Easy to implement
- ✅ Good for interactive applications

**Cons:**
- ⚠️ Subject to OpenAI rate limits (varies by tier)
- ⚠️ Uses standard API pricing (no discount)
- ⚠️ May hit rate limits with too many concurrent requests

**Implementation:**
I've created `compare_llava_models_with_batch.py` which implements this approach.

**How to use:**
```python
# Set batch size (number of concurrent requests)
openai_batch_size = 5  # Adjust based on your rate limits

# The script will automatically use concurrent processing for OpenAI models
python compare_llava_models_with_batch.py
```

**Rate Limits:**
- **Free tier:** 3 requests/minute
- **Tier 1 ($5+):** 500 requests/minute
- **Tier 2 ($50+):** 5,000 requests/minute
- **Tier 3 ($500+):** 10,000 requests/minute

**Recommended batch size:**
- Free tier: 1-2 concurrent requests
- Tier 1: 5-10 concurrent requests
- Tier 2: 10-20 concurrent requests
- Tier 3: 20-50 concurrent requests

---

## Option 2: OpenAI Batch API (Recommended for Large-Scale Processing)

**What it is:** OpenAI's official Batch API for asynchronous processing of large volumes.

**Pros:**
- ✅ **50% cost discount** on both input and output tokens
- ✅ No rate limits (process millions of requests)
- ✅ Better for large-scale processing
- ✅ Results stored in files for easy retrieval

**Cons:**
- ❌ Asynchronous (not real-time, takes hours to complete)
- ❌ More complex setup (requires JSONL file preparation)
- ❌ Results come back later (not immediate)

**How it works:**
1. Prepare a JSONL file with all your requests
2. Upload to OpenAI Batch API
3. Wait for processing (can take hours)
4. Download results file

**Example JSONL format:**
```jsonl
{"custom_id": "request-1", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "..."}]}}
{"custom_id": "request-2", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "..."}]}}
```

**Implementation:**
```python
from openai import OpenAI

client = OpenAI()

# 1. Create JSONL file with requests
with open("batch_requests.jsonl", "w") as f:
    for request in requests:
        f.write(json.dumps(request) + "\n")

# 2. Upload batch file
with open("batch_requests.jsonl", "rb") as f:
    batch_file = client.files.create(
        file=f,
        purpose="batch"
    )

# 3. Create batch job
batch = client.batches.create(
    input_file_id=batch_file.id,
    endpoint="/v1/chat/completions",
    completion_window="24h"
)

# 4. Check status
batch_status = client.batches.retrieve(batch.id)
print(f"Status: {batch_status.status}")  # validating, in_progress, finalizing, completed, expired, cancelled, failed

# 5. Download results when complete
if batch_status.status == "completed":
    output_file_id = batch_status.output_file_id
    # Download and process results
```

---

## Comparison: Concurrent vs Batch API

| Feature | Concurrent Requests | Batch API |
|---------|-------------------|-----------|
| **Speed** | Real-time (seconds) | Asynchronous (hours) |
| **Cost** | Standard pricing | 50% discount |
| **Rate Limits** | Yes (varies by tier) | No limits |
| **Use Case** | Interactive, real-time | Large-scale, offline |
| **Complexity** | Simple | More complex |
| **Best For** | Testing, small batches | Production, large volumes |

---

## Recommended Approach for Your Use Case

### For Testing/Comparison (Current Use Case):
**Use: Concurrent Requests (Option 1)**

- You're testing 6 seconds of video
- Need results quickly for comparison
- Small volume doesn't justify Batch API complexity
- Use `compare_llava_models_with_batch.py` with `batch_size=5`

### For Production (Large-Scale Processing):
**Use: Batch API (Option 2)**

- Processing entire videos (hundreds/thousands of seconds)
- Cost savings matter (50% discount)
- Can wait for results (asynchronous)
- Worth the setup complexity

---

## Implementation in Your Code

### Current Implementation (Sequential):
```python
# Processes one at a time - SLOW
for second_idx in seconds_to_test:
    result = process_with_openai(...)  # Wait for each one
```

### With Concurrent Processing (Option 1):
```python
# Processes multiple in parallel - FAST
with ThreadPoolExecutor(max_workers=5) as executor:
    futures = {executor.submit(process_with_openai, task): task for task in tasks}
    for future in as_completed(futures):
        result = future.result()  # Get results as they complete
```

### Speed Improvement:
- **Sequential:** 6 seconds × 9.73s = **58.4 seconds**
- **Concurrent (5 workers):** ~9.73s + overhead = **~15-20 seconds**
- **Speedup: ~3x faster**

---

## Rate Limit Considerations

### Check Your Rate Limits:
```python
# You can check your rate limits via API
from openai import OpenAI
client = OpenAI()

# Rate limits are shown in API responses
# Or check your OpenAI dashboard
```

### Handle Rate Limits:
```python
import time
from openai import RateLimitError

def process_with_retry(...):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            return process_with_openai(...)
        except RateLimitError as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"Rate limit hit, waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise
```

---

## Example: Using the Batch Processing Script

```bash
# Set your API key
export OPENAI_API_KEY="your-key-here"

# Run with default batch size (5 concurrent requests)
python compare_llava_models_with_batch.py

# Or modify the script to change batch_size:
# openai_batch_size = 10  # For higher tier accounts
```

---

## Summary

**For your current testing use case:**
1. ✅ Use **Concurrent Requests** (Option 1)
2. ✅ Use the provided `compare_llava_models_with_batch.py` script
3. ✅ Set `batch_size=5` (adjust based on your rate limits)
4. ✅ Expect **~3x speedup** over sequential processing

**For production:**
1. ✅ Use **Batch API** (Option 2) for 50% cost savings
2. ✅ Process large volumes asynchronously
3. ✅ No rate limit concerns

The concurrent approach is already implemented in `compare_llava_models_with_batch.py` - just run it!

