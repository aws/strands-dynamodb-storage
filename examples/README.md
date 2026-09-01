# Examples

Runnable, self-contained examples for strands-dynamodb-storage. Every example
was executed against real Amazon DynamoDB before it landed here, and the
sample output in each README is the actual run. Each directory stands alone:
a walkthrough README, one script, and nothing shared, so you can copy a
directory into your own project and start from there.

## Start here

| | Example | What it shows | The DynamoDB angle |
|---|---|---|---|
| 🔄 | [session-resume](./session-resume/) | A conversation survives a restart: a brand-new process picks up the thread from a persisted snapshot | Point reads on the hot path of every agent turn |
| 🧠 | [semantic-memory](./semantic-memory/) | Long-term memory recalled by meaning, with the `MemoryStore` bridge class and Memory Manager wiring | A vector index on the same table as the agent's state |
| 🏢 | [multi-tenant](./multi-tenant/) | Two users share a table; listings and searches provably never cross the boundary | Prefix-scoped keys and partition-scoped vector search |
| ⏳ | [ttl-sessions](./ttl-sessions/) | Ephemeral state that expires itself, with no cleanup job to write | Native TTL: stamp on write, filter on read, reap in the background |
| 📦 | [context-offload](./context-offload/) | A 1 MB value round-trips through the same four-method contract as a 26-byte one | Transparent S3 spillover behind a pointer item |
| 🎧 | [customer-support](./customer-support/) | The capstone: session resume, semantic recall, and tenant isolation combined in one assistant | One table carrying everything the agent remembers |

If you are new to the package, run them in that order: each example
introduces one capability, and the capstone assembles them the way a real
assistant uses them.

## Picking by problem

- "My agent forgets everything on redeploy" → [session-resume](./session-resume/)
- "My agent can't recall what the user said last month" → [semantic-memory](./semantic-memory/)
- "One table, many customers, zero crossover" → [multi-tenant](./multi-tenant/)
- "Anonymous sessions are piling up in my table" → [ttl-sessions](./ttl-sessions/)
- "A tool result blew the 400 KB item limit" → [context-offload](./context-offload/)
- "Show me all of it working together" → [customer-support](./customer-support/)

## Running the examples

Every example needs a DynamoDB table with string keys `pk` and `sk`; the ones
that search memories also need a vector index, and each README opens with the
exact `create-table` call it expects. Credentials follow the least-privilege
policy in the [repository README](../README.md#provisioning-and-permissions).
The scripts come in both languages: each directory carries a Python script
and a TypeScript mirror side by side, running the same pattern against the
same table.

```bash
pip install strands-agents strands-dynamodb-storage
cd session-resume
python session_resume.py --table agent-storage tell "My rental car is a blue Nissan Micra"
```

```bash
npm install @strands-agents/sdk strands-dynamodb-storage tsx
cd session-resume
npx tsx session-resume.ts --table agent-storage tell "My rental car is a blue Nissan Micra"
```

Examples that invoke a model use Amazon Bedrock: the agent examples run on
your configured default model, and embeddings use Amazon Titan Text
Embeddings V2 (1024 dimensions, matching the vector index in each README).

## Contributing an example

Keep the shape: one directory, one README that a reader can follow without
running anything, one script that runs end to end against real DynamoDB, and
no imports from sibling examples. If it shows something the table above
doesn't already cover, we'd love to see it.
