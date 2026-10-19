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
agent model and Titan embeddings (the DynamoDBMemoryStore default embedder).

Usage:
  python support_assistant.py --table agent-storage seed
  python support_assistant.py --table agent-storage ticket \
      "Our uploads keep failing after about a minute. Nothing changed on our side."
  python support_assistant.py --table agent-storage ticket "Thanks. What did we conclude?"
"""

import argparse
import asyncio

from strands import Agent
from strands.memory import MemoryManager
from strands.session import SnapshotSessionManager
from strands_dynamodb_storage import DynamoDBMemoryStore, DynamoDBStorage

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


def build_memory_store(storage: DynamoDBStorage) -> DynamoDBMemoryStore:
    return DynamoDBMemoryStore(
        storage,
        name="account-memory",
        description="This customer's account facts and history, searched by meaning",
        max_search_results=3,
    )


def build_agent(table: str, region: str) -> Agent:
    """One storage instance carries the whole assistant for this customer."""
    storage = DynamoDBStorage(table, region_name=region, prefix=CUSTOMER)
    memory = MemoryManager(stores=[build_memory_store(storage)], add_tool_config=True)
    session = SnapshotSessionManager(TICKET_SESSION, storage=storage)
    return Agent(system_prompt=SYSTEM_PROMPT, session_manager=session, memory_manager=memory)


async def seed(table: str, region: str) -> None:
    store = build_memory_store(DynamoDBStorage(table, region_name=region, prefix=CUSTOMER))
    for fact in ACCOUNT_FACTS:
        await store.add(fact, {"kind": "account-fact"})
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
