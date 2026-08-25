# Customer support assistant

The capstone for this examples library: session persistence, long-term
memory, and tenant isolation combined the way a real assistant uses them,
with one DynamoDB table carrying all of it.

## How it works

One `DynamoDBStorage` instance, constructed with the customer's prefix,
feeds both SDK managers:

```python
storage = DynamoDBStorage(table, prefix="customer/acme")
session = SnapshotSessionManager("ticket-7341", storage=storage)
memory = MemoryManager(stores=[DynamoDBMemoryStore(storage, ...)], add_tool_config=True)
agent = Agent(system_prompt=..., session_manager=session, memory_manager=memory)
```

- The session manager snapshots the ticket conversation, so any process can
  resume it: a redeploy, a handoff, tomorrow's follow-up.
- The memory manager recalls the customer's account facts by meaning before
  each model call, and `add_memory` lets the agent store new conclusions.
- The constructor prefix and partition-scoped search keep every read, write,
  and recall inside this customer's key space. Another customer's assistant
  built the same way cannot reach any of it.

## Run it

Provision the vector-indexed table from the
[semantic-memory example](../semantic-memory/README.md#run-it), then seed the
customer's account facts:

```bash
pip install strands-agents strands-dynamodb-storage
python support_assistant.py --table agent-storage seed
```

Open a ticket. The symptom shares no words with any stored fact:

```bash
python support_assistant.py --table agent-storage ticket \
  "Our uploads keep failing after about a minute. Nothing changed on our side."
```

```text
Tool #1: search_memory
I found a likely culprit right away. Based on your account history, Acme's
network runs an outbound proxy that closes idle connections after 60
seconds. That aligns perfectly with uploads failing "after about a minute."
...
And since you prefer email communication, I can send a detailed write-up...
```

The agent connected "failing after about a minute" to a proxy memory it
retrieved by meaning, folded in the pinned SDK version, and honored the
contact preference, all from DynamoDB, none of it in the prompt.

Come back to the ticket from a completely new process:

```bash
python support_assistant.py --table agent-storage ticket \
  "Thanks. Summarize in one sentence what we concluded the cause was on this ticket."
```

```text
The likely cause of your upload failures is Acme's outbound proxy closing
idle connections after 60 seconds, which interrupts uploads that have brief
idle gaps during data transfer.
```

The conclusion came from the resumed session snapshot; the new process
started with nothing in memory.

## What each example contributed

| Capability | Shown alone in |
|---|---|
| Session resume across processes | [session-resume](../session-resume/) |
| Recall by meaning on a vector index | [semantic-memory](../semantic-memory/) |
| Per-customer isolation | [multi-tenant](../multi-tenant/) |
| Self-expiring state | [ttl-sessions](../ttl-sessions/) |
| Oversized values to S3 | [context-offload](../context-offload/) |

Add `ttl_seconds` to the constructor to expire closed tickets, and
`s3_bucket` to absorb oversized tool results; both compose with everything
above without touching the agent code.

## Clean up

```bash
aws dynamodb delete-table --table-name agent-storage
```
