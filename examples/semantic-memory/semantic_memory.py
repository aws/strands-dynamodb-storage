"""Give an agent long-term memory it can search by meaning, on one DynamoDB table.

Session persistence gets a conversation through a restart. This example covers
the other half: recalling something a user said weeks ago, in a new
conversation that shares no keys or words with the old one. DynamoDBMemoryStore
embeds each memory with Amazon Titan Text Embeddings V2, stores it with its
vector on the table, and recalls it with a vector index search scoped to the
user's partition, so the Memory Manager can pull relevant memories into the
model input before each call.

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

from strands import Agent
from strands.memory import MemoryManager
from strands_dynamodb_storage import DynamoDBMemoryStore, DynamoDBStorage

USER_NAMESPACE = "user/u1"

SEED_MEMORIES = [
    "Prefers window seats on long flights",
    "Planning a trip to Tokyo in December",
    "Has a peanut allergy, flag it when booking meals",
]


def build_store(table: str, region: str) -> DynamoDBMemoryStore:
    """One memory store per user: the namespace keeps every user's memories in their own partition."""
    storage = DynamoDBStorage(table, region_name=region).namespace(USER_NAMESPACE)
    return DynamoDBMemoryStore(storage, max_search_results=3)


async def seed(store: DynamoDBMemoryStore) -> None:
    for memory in SEED_MEMORIES:
        key = await store.add(memory, {"kind": "preference"})
        print(f"stored: {memory} ({key})")


def ask(store: DynamoDBMemoryStore, question: str) -> None:
    memory = MemoryManager(stores=[store], add_tool_config=True)
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
