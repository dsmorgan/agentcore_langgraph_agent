# AgentCore Observability - CloudWatch Logs Insights Queries

This document contains ready-to-use CloudWatch Logs Insights queries for monitoring your AgentCore LangGraph agent.

## Quick Start

After deploying with `agentcore launch`, your logs will be in CloudWatch Logs at:
- Log group: `/aws/bedrock/agentcore/agent-rZ3MAJDqLa`

## Essential Queries

### 1. End-to-End Latency Analysis

Monitor overall invocation performance by streaming vs non-streaming mode.

```
fields @timestamp, session_id, actor_id, duration_ms, stream_mode
| filter operation = "invocation_complete"
| stats avg(duration_ms) as avg_latency, max(duration_ms) as max_latency, count() as invocations by stream_mode
```

**Use case**: Track response times and compare streaming vs non-streaming performance.

---

### 2. Memory Performance Monitoring

Track memory retrieval and save operations performance.

```
fields @timestamp, session_id, operation, duration_ms, memory_count
| filter operation in ["memory_retrieval", "memory_save"]
| stats avg(duration_ms) as avg_duration, max(duration_ms) as max_duration, count() as operations by operation
```

**Use case**: Identify slow memory operations and optimize if needed.

---

### 3. Tool Usage Analytics

See which tools are used most frequently and their performance.

```
fields @timestamp, session_id, tool_name, duration_ms
| filter operation = "tool_end"
| stats count() as executions, avg(duration_ms) as avg_duration, max(duration_ms) as max_duration by tool_name
| sort executions desc
```

**Use case**: Understand tool usage patterns and identify slow tools.

---

### 4. Error Analysis

Find and categorize errors by operation type.

```
fields @timestamp, session_id, actor_id, operation, error_type, message
| filter error_type != ""
| stats count() as error_count by operation, error_type
| sort error_count desc
```

**Use case**: Debug issues and track error rates.

---

### 5. Session Timeline

Reconstruct the complete timeline of a specific session for debugging.

```
fields @timestamp, operation, duration_ms, tool_name, memory_count
| filter session_id = "YOUR_SESSION_ID"
| sort @timestamp asc
```

**Use case**: Debug specific user sessions by seeing all operations in order.

---

### 6. Actor Activity

See which users are using the agent most.

```
fields @timestamp, actor_id, session_id, operation
| filter operation = "session_start"
| stats count() as sessions by actor_id
| sort sessions desc
```

**Use case**: Track user engagement and identify power users.

---

### 7. Memory Hit Rate

Calculate how often long-term memory returns results.

```
fields @timestamp, memory_count
| filter operation = "memory_retrieval"
| stats count() as total_retrievals, sum(memory_count > 0) as hits
| extend hit_rate = 100 * hits / total_retrievals
```

**Use case**: Understand memory effectiveness and data retention.

---

### 8. P95 Latency by Operation

Identify slowest operations using 95th percentile.

```
fields operation, duration_ms
| filter duration_ms > 0
| stats pct(duration_ms, 95) as p95_latency by operation
| sort p95_latency desc
```

**Use case**: Find performance bottlenecks in specific operations.

---

### 9. Recent Errors (Last 1 Hour)

Quick view of recent errors for troubleshooting.

```
fields @timestamp, session_id, operation, error_type, message
| filter error_type != ""
| sort @timestamp desc
| limit 50
```

**Use case**: Real-time error monitoring and alerting.

---

### 10. Daily Invocation Summary

Get daily statistics on agent usage.

```
fields @timestamp, operation, stream_mode
| filter operation = "invocation_complete"
| stats count() as total_invocations,
        sum(stream_mode) as streaming_invocations,
        sum(stream_mode == false) as non_streaming_invocations
```

**Use case**: Track overall usage trends and patterns.

---

## Built-in AgentCore Metrics

In addition to custom logs, AgentCore automatically provides these CloudWatch metrics:

### Accessing via AWS Console

1. Go to: **CloudWatch > GenAI Observability > Bedrock AgentCore**
2. Agent ID: `agent-rZ3MAJDqLa`

### Available Metrics (AWS/BedrockAgentCore namespace)

- **Invocations** - Total number of agent invocations
- **InvocationLatency** - P50, P90, P99 latency percentiles
- **Errors** - Error count and error rate
- **TokenCount** - Input and output token usage
- **ToolExecutions** - Tool usage count

### Memory Metrics (AWS/BedrockAgentCore/Memory namespace)

- **MemoryStoreLatency** - Latency for STM/LTM writes
- **MemoryRetrievalLatency** - Latency for STM/LTM reads
- **MemoryStoreCount** - Number of memory stores
- **MemoryRetrievalCount** - Number of memory retrievals

### CLI Example

```bash
# Get invocation count for the last 24 hours
aws cloudwatch get-metric-statistics \
  --namespace AWS/BedrockAgentCore \
  --metric-name Invocations \
  --dimensions Name=AgentId,Value=agent-rZ3MAJDqLa \
  --statistics Sum \
  --start-time $(date -u -d '24 hours ago' +%Y-%m-%dT%H:%M:%S) \
  --end-time $(date -u +%Y-%m-%dT%H:%M:%S) \
  --period 3600
```

---

## Setting Up Alarms

### High Error Rate Alarm

```bash
aws cloudwatch put-metric-alarm \
  --alarm-name agentcore-high-error-rate \
  --alarm-description "Alert when error rate exceeds 5%" \
  --metric-name Errors \
  --namespace AWS/BedrockAgentCore \
  --statistic Sum \
  --period 300 \
  --evaluation-periods 2 \
  --threshold 5 \
  --comparison-operator GreaterThanThreshold \
  --dimensions Name=AgentId,Value=agent-rZ3MAJDqLa
```

### High Latency Alarm

```bash
aws cloudwatch put-metric-alarm \
  --alarm-name agentcore-high-latency \
  --alarm-description "Alert when P95 latency exceeds 5 seconds" \
  --metric-name InvocationLatency \
  --namespace AWS/BedrockAgentCore \
  --statistic Average \
  --period 300 \
  --evaluation-periods 2 \
  --threshold 5000 \
  --comparison-operator GreaterThanThreshold \
  --dimensions Name=AgentId,Value=agent-rZ3MAJDqLa
```

---

## Operations Reference

Your agent logs these operation types:

| Operation | Description | Key Fields |
|-----------|-------------|------------|
| `session_start` | New invocation started | `session_id`, `actor_id`, `stream_mode` |
| `memory_retrieval` | Long-term memory search | `memory_count`, `duration_ms` |
| `memory_save` | Conversation saved to memory | `duration_ms` |
| `tool_start` | Tool execution began | `tool_name` |
| `tool_end` | Tool execution completed | `tool_name`, `duration_ms` |
| `invocation_complete` | Request finished | `duration_ms`, `stream_mode` |

## Cost Optimization Tips

1. **Use appropriate time ranges** - Limit queries to necessary time windows
2. **Filter early** - Put filter clauses before stats to reduce scanned data
3. **Sample data** - Use `| sample 1000` for exploratory queries
4. **Archive old logs** - Configure lifecycle policies to archive to S3

## Troubleshooting

### No logs appearing?

1. Check log group exists:
   ```bash
   aws logs describe-log-groups --log-group-name-prefix /aws/bedrock/agentcore
   ```

2. Verify agent is deployed:
   ```bash
   agentcore status
   ```

3. Check IAM permissions for CloudWatch Logs write access

### Missing structured fields?

- Ensure you're running the latest version of agent.py with StructuredFormatter
- Verify JSON format: `aws logs tail /aws/bedrock/agentcore/agent-rZ3MAJDqLa --format json`

---

## Additional Resources

- [AWS CloudWatch Logs Insights Query Syntax](https://docs.aws.amazon.com/AmazonCloudWatch/latest/logs/CWL_QuerySyntax.html)
- [AgentCore Observability Guide](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/observability.html)
- [Plan Document](/Users/dsm/.claude/plans/staged-giggling-kahan.md) - Full implementation details
