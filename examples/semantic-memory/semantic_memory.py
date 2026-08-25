"""Give an agent long-term memory it can search by meaning, on one DynamoDB table.

Session persistence gets a conversation through a restart. This example covers
the other half: recalling something a user said weeks ago, in a new
conversation that shares no keys or words with the old one. Memories are
embedded on write, stored with a vector alongside the bytes, and recalled with
a DynamoDB vector index search scoped to the user's partition.

The ``DynamoDBMemoryStore`` class here is application code: the package ships
the storage contract and vector search, and this bridge adapts them to the
Strands SDK's ``MemoryStore`` protocol so the Memory Manager can pull relevant
memories into the model input before each call.

Prerequisites:
  - A DynamoDB table with a vector index (see README.md; 1024 dims, COSINE).
  - Credentials for the table, dynamodb:SearchVectors, and Bedrock InvokeModel
    (agent model + amazon.titan-embed-text-v2:0 for embeddings).

Usage:
  python semantic_memory.py --table agent-storage seed
  python semantic_memory.py --table agent-storage ask \
      "Book me a flight seat for my December trip. Which seat should I pick?"
"""

import argparse
import asyncio
import json
import uuid
from typing import Any, Optional

import boto3
from strands import Agent
from strands.memory import MemoryEntry, MemoryManager
from strands.memory.types import Metadata, SearchOptions
from strands_dynamodb_storage import DynamoDBStorage, SearchQuery

USER_PARTITION = "user/u1"

SEED_MEMORIES = [
    "Prefers window seats on long flights",
    "Planning a trip to Tokyo in December",
    "Has a peanut allergy, flag it when booking meals",
]


def make_embedder(region: str) -> Any:
    """Return an embed(text) -> list[float] callable backed by Titan V2 (1024 dims)."""
    bedrock = boto3.client("bedrock-runtime", region_name=region)

    def embed(text: str) -> list[float]:
        response = bedrock.invoke_model(
            modelId="amazon.titan-embed-text-v2:0",
            body=json.dumps({"inputText": text, "dimensions": 1024}),
        )
        return json.loads(response["body"].read())["embedding"]  # type: ignore[no-any-return]

    return embed


class DynamoDBMemoryStore:
    """Adapts DynamoDBStorage vector search to the Strands MemoryStore protocol."""

    def __init__(self, storage: DynamoDBStorage, partition: str, embed: Any) -> None:
        self.storage = storage
        self.partition = partition
        self.embed = embed
        self.name = "dynamodb"
        self.description = "Long-term memories in DynamoDB, searched by meaning"
        self.max_search_results = 3
        self.writable = True
        self.extraction = None

    async def add(self, content: str, metadata: Optional[Metadata] = None) -> None:
        await self.storage.write(
            f"memories/{uuid.uuid4().hex[:8]}",
            content.encode(),
            vector=self.embed(content),
            metadata=metadata,
        )

    async def search(self, query: str, options: Optional[SearchOptions] = None) -> list[MemoryEntry]:
        results = await self.storage.search(
            SearchQuery(
                vector=self.embed(query),
                top_k=self.max_search_results,
                pk=self.partition,
                include_values=True,
            )
        )
        return [
            MemoryEntry(content=r.data.decode(), metadata=r.metadata)
            for r in results
            if r.data is not None
        ]


def build_store(table: str, region: str) -> DynamoDBMemoryStore:
    storage = DynamoDBStorage(table, region_name=region, prefix=USER_PARTITION)
    return DynamoDBMemoryStore(storage, partition=USER_PARTITION, embed=make_embedder(region))


async def seed(store: DynamoDBMemoryStore) -> None:
    for memory in SEED_MEMORIES:
        await store.add(memory, metadata={"kind": "preference"})
        print(f"stored: {memory}")


def ask(store: DynamoDBMemoryStore, question: str) -> None:
    # MemoryStore's optional methods (add_messages, initialize, get_tools) are
    # detected at runtime by the manager, but mypy requires them structurally.
    memory = MemoryManager(stores=[store], add_tool_config=True)  # type: ignore[list-item]
    agent = Agent(memory_manager=memory)
    agent(question)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", default="agent-storage")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("verb", choices=["seed", "ask"])
    parser.add_argument("question", nargs="?", default="Which seat should I pick for my December trip?")
    args = parser.parse_args()

    store = build_store(args.table, args.region)
    if args.verb == "seed":
        asyncio.run(seed(store))
    else:
        ask(store, args.question)


if __name__ == "__main__":
    main()
