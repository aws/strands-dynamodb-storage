"""Resume a conversation after a restart, with session state in Amazon DynamoDB.

Run 1 tells the agent a fact and exits, taking the process (and all in-memory
state) with it. Run 2 constructs a brand-new agent with the same session id and
asks about the fact. The answer comes from the snapshot that
``SnapshotSessionManager`` persisted to DynamoDB, not from anything held in
memory.

Prerequisites:
  - A DynamoDB table with string keys ``pk`` / ``sk`` (see README.md).
  - AWS credentials with PutItem/GetItem/DeleteItem/Query on the table and
    InvokeModel on your Bedrock model.

Usage:
  python session_resume.py --table agent-storage tell "My rental car is a blue Nissan Micra"
  python session_resume.py --table agent-storage ask "What car am I driving?"
"""

import argparse

from strands import Agent
from strands.session import SnapshotSessionManager
from strands_dynamodb_storage import DynamoDBStorage

SESSION_ID = "session-resume-demo"


def build_agent(table: str, region: str) -> Agent:
    """Construct an agent whose session state lives in DynamoDB.

    A new process starts with no in-memory state; if a snapshot for
    SESSION_ID exists in the table, the session manager restores it here.
    """
    storage = DynamoDBStorage(table, region_name=region)
    session = SnapshotSessionManager(SESSION_ID, storage=storage)
    return Agent(session_manager=session)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", default="agent-storage")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("verb", choices=["tell", "ask"])
    parser.add_argument("message")
    args = parser.parse_args()

    agent = build_agent(args.table, args.region)
    agent(args.message)


if __name__ == "__main__":
    main()
