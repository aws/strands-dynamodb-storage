"""Self-expiring agent state with DynamoDB-native TTL.

Ephemeral state accumulates fast in agent systems: anonymous visitor
sessions, one-off tool scratch space, verification codes. Without a retention
mechanism, someone ends up writing a cleanup job. With this package,
retention is a constructor argument: every write stamps an epoch-seconds
``expireAt`` attribute, reads and listings filter items whose expiry has
passed, and DynamoDB physically reaps them in the background at no request
cost.

The script writes one item with a short TTL and one without, waits out the
expiry, and shows all three views of the aftermath: the expired item is gone
from ``read`` and ``list``, the persistent item is untouched, and the raw
DynamoDB item may still be physically present for a while, which is why the
package filters on read rather than trusting the reaper's timing.

Prerequisites:
  - A pk/sk table with TTL enabled on ``expireAt`` (see README.md).
  - Credentials with PutItem/GetItem/DeleteItem/Query on the table.

Usage:
  python ttl_sessions.py --table agent-storage
"""

import argparse
import asyncio

import boto3
from strands_dynamodb_storage import DynamoDBStorage

TTL_SECONDS = 8


async def main_async(table: str, region: str) -> None:
    # ttl_seconds opts the store in to TTL: stamp on write, filter on read/list.
    ephemeral = DynamoDBStorage(table, region_name=region, prefix="guest/g1", ttl_seconds=TTL_SECONDS)
    durable = DynamoDBStorage(table, region_name=region, prefix="guest/g1")

    await ephemeral.write("session/cart", b"3 items, checkout not started")
    await durable.write("profile/consent", b"cookie banner accepted v3")
    print(f"wrote session/cart with ttl_seconds={TTL_SECONDS}, profile/consent with no TTL")

    data = await ephemeral.read("session/cart")
    print(f"before expiry: read session/cart -> {data!r}")
    print(f"before expiry: list ->  {sorted(await ephemeral.list('session/') + await durable.list('profile/'))}")

    print(f"\nwaiting {TTL_SECONDS + 2}s for expiry...")
    await asyncio.sleep(TTL_SECONDS + 2)

    # The store filters the expired item immediately; DynamoDB reaps it later.
    print(f"after expiry: read session/cart -> {await ephemeral.read('session/cart')}")
    print(f"after expiry: list session/ -> {await ephemeral.list('session/')}")
    print(f"after expiry: read profile/consent -> {await durable.read('profile/consent')}")

    # Peek underneath: the raw item can still be physically present until the
    # background reaper removes it. This is why the package filters on read.
    ddb = boto3.client("dynamodb", region_name=region)
    raw = ddb.get_item(TableName=table, Key={"pk": {"S": "guest/g1"}, "sk": {"S": "session/cart"}})
    print(f"raw GetItem still returns an item: {'Item' in raw} (reaper removes it in the background)")

    await durable.delete("profile/consent")
    print("\ndemo items cleaned up (expired item left for the reaper)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", default="agent-storage")
    parser.add_argument("--region", default="us-east-1")
    args = parser.parse_args()
    asyncio.run(main_async(args.table, args.region))


if __name__ == "__main__":
    main()
