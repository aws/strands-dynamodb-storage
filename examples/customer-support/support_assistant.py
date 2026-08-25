"""A support assistant with per-customer memory, on one DynamoDB table.

The capstone for this examples library: everything the other examples show,
combined the way a real assistant uses it. One table carries, per customer:

- **Session state**, so a ticket conversation survives restarts and handoffs
  (``SnapshotSessionManager``).
- **Long-term memories**, recalled by meaning across tickets
  (``MemoryManager`` over a DynamoDB vector index).
- **Tenant isolation**, so one customer's history never reaches another's
  conversation (constructor prefix + partition-scoped search).

The demo seeds account facts for one customer, then opens a support ticket
in a fresh process: the customer reports dropped connections, and the agent
connects that to a months-old memory about their proxy closing idle sockets,
which shares no words with the complaint. A follow-up run resumes the same
ticket from the persisted session.

Prerequisites: vector-indexed table (see README.md), Bedrock access for the
agent model and Titan embeddings.

Usage:
  python support_assistant.py --table agent-storage seed
  python support_assistant.py --table agent-storage ticket \
      "Our uploads keep failing after about a minute. Nothing changed on our side."
  python support_assistant.py --table agent-storage ticket "Thanks. What did we conclude?"
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
from strands.session import SnapshotSessionManager
from strands_dynamodb_storage import DynamoDBStorage, SearchQuery

CUSTOMER = "customer/acme"
TICKET_SESSION = "ticket-7341"

ACCOUNT_FACTS = [
    "Acme's network runs an outbound proxy that closes idle connections after 60 seconds",
    "Acme pinned their integration to a custom build of the SDK, version 2.14",
    "Acme's technical contact prefers email and asked not to be called by phone",
]

SYSTEM_PROMPT = (
    "You are a support engineer for a data-transfer service. Use the customer's "
    "account memories when they are relevant to the reported symptom, and say "
    "which remembered fact you are basing your answer on."
)


def make_embedder(region: str) -> Any:
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
        self.name = "account-memory"
        self.description = "This customer's account facts and history, searched by meaning"
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


def build_agent(table: str, region: str) -> Agent:
    """One storage instance carries the whole assistant for this customer."""
    storage = DynamoDBStorage(table, region_name=region, prefix=CUSTOMER)
    store = DynamoDBMemoryStore(storage, partition=CUSTOMER, embed=make_embedder(region))
    # MemoryStore's optional methods are detected at runtime by the manager,
    # but mypy requires them structurally.
    memory = MemoryManager(stores=[store], add_tool_config=True)  # type: ignore[list-item]
    session = SnapshotSessionManager(TICKET_SESSION, storage=storage)
    return Agent(system_prompt=SYSTEM_PROMPT, session_manager=session, memory_manager=memory)


async def seed(table: str, region: str) -> None:
    storage = DynamoDBStorage(table, region_name=region, prefix=CUSTOMER)
    store = DynamoDBMemoryStore(storage, partition=CUSTOMER, embed=make_embedder(region))
    for fact in ACCOUNT_FACTS:
        await store.add(fact, metadata={"kind": "account-fact"})
        print(f"stored: {fact}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", default="agent-storage")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("verb", choices=["seed", "ticket"])
    parser.add_argument("message", nargs="?", default="Our uploads keep failing after about a minute.")
    args = parser.parse_args()

    if args.verb == "seed":
        asyncio.run(seed(args.table, args.region))
    else:
        agent = build_agent(args.table, args.region)
        agent(args.message)


if __name__ == "__main__":
    main()
