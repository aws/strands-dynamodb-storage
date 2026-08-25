"""Store values beyond the DynamoDB item limit with transparent S3 offload.

Agent workloads produce oversized values routinely: a tool returns a whole
web page, a session accumulates a long transcript, a context offloader parks
a large intermediate result. DynamoDB items cap at 400 KB. Configuring the
store with an S3 bucket makes that cap invisible: values above the limit are
offloaded to S3 with a small pointer item remaining in the table, and
``read`` returns the full bytes either way. Callers see one contract
regardless of payload size.

The script writes a small value and a 1 MB value through the same store,
reads both back, then peeks underneath to show where each physically lives:
the small value inline in its DynamoDB item, the large one as an S3 object
behind a pointer item. Deleting the key reclaims the S3 object too.

Prerequisites:
  - A pk/sk table (see README.md) and an S3 bucket you own.
  - Credentials for the table plus s3:PutObject/GetObject/DeleteObject on
    the bucket.

Usage:
  python context_offload.py --table agent-storage --bucket my-offload-bucket
"""

import argparse
import asyncio

import boto3
from strands_dynamodb_storage import DynamoDBStorage

LARGE_SIZE = 1_000_000  # 1 MB, over the 400 KB item limit


async def main_async(table: str, bucket: str, region: str) -> None:
    storage = DynamoDBStorage(table, region_name=region, prefix="session/s1", s3_bucket=bucket)

    small = b"user asked about seat maps"
    large = b"<html>" + b"tool result far too big for one item " * 28000 + b"</html>"
    assert len(large) > LARGE_SIZE

    await storage.write("turns/t1", small)
    await storage.write("turns/t2-page", large)
    print(f"wrote {len(small)} B and {len(large):,} B through the same contract")

    # Reads are symmetric: full bytes come back either way.
    r1 = await storage.read("turns/t1")
    r2 = await storage.read("turns/t2-page")
    assert r1 == small and r2 == large
    print(f"read back {len(r1 or b'')} B and {len(r2 or b''):,} B, both intact")
    print(f"list sees both: {sorted(await storage.list('turns/'))}\n")

    # Peek underneath at where each value physically lives.
    ddb = boto3.client("dynamodb", region_name=region)
    s3 = boto3.client("s3", region_name=region)
    for sk in ("turns/t1", "turns/t2-page"):
        item = ddb.get_item(TableName=table, Key={"pk": {"S": "session/s1"}, "sk": {"S": sk}})["Item"]
        size = len(item.get("data", {}).get("B", b""))
        print(f"{sk}: DynamoDB item carries {size} B of data" + ("" if size else " (pointer item)"))
    objects = s3.list_objects_v2(Bucket=bucket).get("Contents", [])
    print(f"S3 objects in bucket: {[(o['Key'], o['Size']) for o in objects]}\n")

    # Delete reclaims the S3 object along with the item.
    await storage.delete("turns/t2-page")
    await storage.delete("turns/t1")
    remaining = s3.list_objects_v2(Bucket=bucket).get("KeyCount", 0)
    print(f"after delete: S3 objects remaining: {remaining}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", default="agent-storage")
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--region", default="us-east-1")
    args = parser.parse_args()
    asyncio.run(main_async(args.table, args.bucket, args.region))


if __name__ == "__main__":
    main()
