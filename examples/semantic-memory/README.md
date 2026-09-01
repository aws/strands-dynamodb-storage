# Semantic long-term memory

Session persistence gets a conversation through a restart. The harder problem
is recalling something a user said weeks ago, in a new conversation that
shares no keys or words with the old one. This example gives a Strands agent
that recall using a DynamoDB vector index on the same table that holds the
rest of its state: memories are embedded on write, and recalled by meaning at
question time.

## How it works

The package ships the storage contract and vector search
(`write(vector=...)` and `search(SearchQuery)`); the Strands SDK's Memory
Manager consumes anything implementing its `MemoryStore` protocol. The bridge
between them is a small class you define in your own application, included
here as `DynamoDBMemoryStore`:

- `add()` embeds the content (Amazon Titan Text Embeddings V2, 1024
  dimensions) and writes bytes + vector + metadata in one call.
- `search()` embeds the query and runs a vector index search scoped to the
  user's partition, returning `MemoryEntry` values.

Wiring it into the agent is one constructor argument:

```python
memory = MemoryManager(stores=[DynamoDBMemoryStore(storage, partition="user/u1", embed=embed)],
                       add_tool_config=True)
agent = Agent(memory_manager=memory)
```

The Memory Manager retrieves relevant entries before each model call and folds
them into the model input, and `add_tool_config=True` registers an
`add_memory` tool so the model can store new facts through the same table it
recalls from.

Notice the `pk` argument in `search()`. The vector index is partitioned the
same way the table is, so every search is scoped to one partition: a user's
search never ranges over another user's memories. The partition value is
supplied by the caller, so this is query scoping rather than an authorization
boundary; access control belongs in AWS IAM and your application layer.

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
