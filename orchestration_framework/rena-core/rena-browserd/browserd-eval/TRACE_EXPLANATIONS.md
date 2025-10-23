# Trace Explanations

The trace explanation feature generates human-readable narratives of execution traces using LLMs, helping developers and users understand what happened during evaluation runs.

## Overview

Trace explanations are generated at two levels:

1. **Per-run explanations**: Describe what happened in each individual evaluation run
2. **Aggregated explanations**: Analyze patterns across multiple runs of the same evaluation

## Configuration

### Using Existing LLM Evaluator

By default, trace explanations will use the first LLM evaluator configuration found in your `eval.toml`:

```toml
[[evaluators]]
type = "llm"
id = "basic-llm"
base_prompt = "..."

[evaluators.inference_engine]
type = "openai"
model = "o4-mini"
max_tokens = 4096
api_key = "<your-api-key>"
```

### Dedicated Trace Explainer Configuration

You can optionally configure a separate LLM specifically for trace explanations:

```toml
[trace_explainer]
type = "anthropic"
model = "claude-sonnet-4-20250514"
max_tokens = 2048
api_key = "<your-anthropic-api-key>"
```

This is useful when you want to:

- Use a different model for explanations (e.g., a more capable model)
- Use different token limits
- Keep evaluation and explanation concerns separate

## How It Works

1. **Post-evaluation processing**: Trace explanations are generated after all evaluation runs complete
2. **Batch processing**: All explanations are generated in a batch to minimize latency impact
3. **Graceful degradation**: If explanation generation fails for any run, the evaluation continues normally

## Example Output

### Per-Run Explanation

```json
{
  "trace_explanation": "The system processed the query 'say hello to Bob' by discovering and launching the hello-world app in 45ms. After listing available tools (8ms), it executed the say_hello tool with parameter 'Bob', successfully generating the response 'Hello, Bob!' in 89ms total using 120 input and 45 output tokens."
}
```

### Aggregated Explanation

```json
{
  "trace_explanation": "Across 5 runs, the hello-world app was consistently selected as the top match with minimal discovery variance (10-12ms). All runs successfully executed the say_hello tool with remarkably stable performance (mean: 85ms, std: 3.2ms). Token usage was highly consistent at ~165 total tokens per run, indicating deterministic behavior. The 100% success rate demonstrates reliable tool discovery and execution."
}
```

## What Gets Explained

The trace explainer analyzes:

- **Execution flow**: Which apps were discovered, what tools were called
- **Performance metrics**: Latencies at each step, token usage
- **Success/failure patterns**: Why evaluations passed or failed
- **Consistency**: Variations across multiple runs
- **Anomalies**: Unusual latencies or unexpected behaviors

## Performance Considerations

- Trace explanations add latency proportional to the number of runs
- Each explanation typically requires 1-2 LLM calls
- For large evaluation suites, consider using a faster model for explanations
- Explanations are generated sequentially to avoid rate limits

## Disabling Trace Explanations

If you don't want trace explanations:

1. Don't configure any LLM evaluators, OR
2. Remove the `[trace_explainer]` section if present

Trace explanations will be skipped with an info log message.

## Integration with Reports

Trace explanations are included in both JSON and CSV report formats:

- JSON: Full explanations in `trace_explanation` fields
- CSV: Explanations are included as additional columns (may be truncated)

## Tips

1. **Use descriptive queries**: Clear queries lead to better explanations
2. **Set appropriate token limits**: Explanations need enough tokens to be comprehensive
3. **Choose the right model**: More capable models produce better explanations
4. **Monitor costs**: Trace explanations add LLM API calls proportional to your evaluation runs
