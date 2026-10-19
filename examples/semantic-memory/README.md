# Semantic long-term memory

Session persistence gets a conversation through a restart. The harder problem
is recalling something a user said weeks ago, in a new conversation that
shares no keys or words with the old one. This example gives a Strands agent
that recall using a DynamoDB vector index on the same table that holds the
rest of its state: memories are embedded on write, and recalled by meaning at
question time.

## How it works

`DynamoDBMemoryStore` implements the Strands SDK's `MemoryStore` protocol, so
the Memory Manager consumes it directly:

- `add()` embeds the content (Amazon Titan Text Embeddings V2, 1024
  dimensions) and writes the text, vector, and metadata as one item.
- `search()` embeds the query and runs one vector index search scoped to the
  user's partition, returning `MemoryEntry` values closest first.

Wiring it into the agent is one constructor argument:

```python
storage = DynamoDBStorage("agent-storage", region_name="us-east-1").namespace("user/u1")
memory = MemoryManager(stores=[DynamoDBMemoryStore(storage, max_search_results=3)], add_tool_config=True)
agent = Agent(memory_manager=memory)
```

The Memory Manager retrieves relevant entries before each model call and folds
them into the model input, and `add_tool_config=True` registers an
`add_memory` tool so the model can store new facts through the same table it
recalls from.

Notice the `namespace("user/u1")` call. The vector index is partitioned the
same way the table is, on the first two segments of each key, and the memory
store searches only the partition that holds its namespace (`user/u1`), so a
user's search never ranges over another user's memories. Keep the user in the
first two segments: `tenant/users/u1` and `tenant/users/u2` would share the
partition `tenant/users`. The namespace is chosen by your application, so this is query
scoping rather than an authorization boundary; access control belongs in AWS
IAM and your application layer.

## Run it

Create a table with a vector index sized to the embedding model (once):

```bash
aws dynamodb create-table \
  --table-name agent-storage \
  --attribute-definitions AttributeName=pk,AttributeType=S AttributeName=sk,AttributeType=S \
  --key-schema AttributeName=pk,KeyType=HASH AttributeName=sk,KeyType=RANGE \
  --billing-mode PAY_PER_REQUEST \
  --vector-indexes '[{
    "IndexName": "vector_index",
    "VectorAttribute": {"AttributeName": "vector"},
    "SearchSchema": [{"AttributeName": "pk", "SearchSchemaElementType": "HASH"}],
    "Projection": {"ProjectionType": "ALL"},
    "Dimensions": 1024,
    "DistanceFunction": "COSINE"
  }]'

pip install strands-agents strands-dynamodb-storage
```

Wait for the index to show `IndexStatus: ACTIVE` with `Backfilling` false in
`describe-table`, then seed three memories and ask:

```bash
python semantic_memory.py --table agent-storage seed
python semantic_memory.py --table agent-storage ask \
  "Book me a flight seat for my December trip. Which seat should I pick?"
```

Or the TypeScript equivalent:

```bash
npm install @strands-agents/sdk strands-dynamodb-storage @aws-sdk/client-bedrock-runtime tsx

npx tsx semantic-memory.ts --table agent-storage seed
npx tsx semantic-memory.ts --table agent-storage ask \
  "Book me a flight seat for my December trip. Which seat should I pick?"
```

```text
Tool #1: search_memory
Based on your saved preferences and trip details, here's what I found:
Trip: Tokyo in December
Seat Preference: You prefer window seats on long flights
...
You have a peanut allergy — make sure to flag that when booking your meal!
```

The question shares almost no words with the stored memories. All three
reached the model by meaning, retrieved from DynamoDB before the model call.

## Why DynamoDB here

The vector index lives on the same table as the agent's sessions and state,
so there is no separate vector database to provision, no pipeline copying
data into it, and no reconciliation job when the two disagree. Vector indexes
are eventually consistent (the same model as a global secondary index), so a
memory written moments ago may take a short time to become searchable.

## Clean up

```bash
aws dynamodb delete-table --table-name agent-storage
```
