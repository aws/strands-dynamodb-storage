# Self-expiring sessions with TTL

Ephemeral state accumulates fast in agent systems: anonymous visitor
sessions, tool scratch space, verification codes. Without a retention
mechanism someone ends up writing a cleanup job. This example makes retention
a constructor argument instead, using DynamoDB-native TTL.

## How it works

Constructing the store with `ttl_seconds` opts it in to TTL:

```python
ephemeral = DynamoDBStorage(table, prefix="guest/g1", ttl_seconds=3600)
```

Three things follow:

- **Every write stamps** an epoch-seconds `expireAt` attribute (a per-write
  `ttl_seconds` can override the default).
- **Reads and listings filter** items whose expiry has already passed, so
  expired data is gone from the application's view immediately.
- **DynamoDB physically reaps** expired items in the background, at no
  request cost, typically within a few days.

The read-time filter is the part that matters for correctness: the reaper's
timing is not a contract, so the package never trusts it. The script makes
this visible by peeking underneath with a raw `GetItem` after expiry: the
item is still physically there, while `read` and `list` already report it
gone.

Stores with and without TTL coexist on one table: the script writes an
ephemeral cart session and a durable consent record under the same tenant
prefix, and only the cart expires.

## Run it

Create the table and enable TTL on the package's expiry attribute (once):

```bash
aws dynamodb create-table \
  --table-name agent-storage \
  --attribute-definitions AttributeName=pk,AttributeType=S AttributeName=sk,AttributeType=S \
  --key-schema AttributeName=pk,KeyType=HASH AttributeName=sk,KeyType=RANGE \
  --billing-mode PAY_PER_REQUEST

aws dynamodb update-time-to-live \
  --table-name agent-storage \
  --time-to-live-specification "Enabled=true, AttributeName=expireAt"

pip install strands-dynamodb-storage boto3
python ttl_sessions.py --table agent-storage
```

```text
wrote session/cart with ttl_seconds=8, profile/consent with no TTL
before expiry: read session/cart -> b'3 items, checkout not started'
before expiry: list ->  ['profile/consent', 'session/cart']

waiting 10s for expiry...
after expiry: read session/cart -> None
after expiry: list session/ -> []
after expiry: read profile/consent -> b'cookie banner accepted v3'
raw GetItem still returns an item: True (reaper removes it in the background)
```

## Considerations

- TTL filtering applies to `read` and `list`. `search()` can briefly return
  an expired item that DynamoDB has not yet physically removed.
- If you combine TTL with S3 offload, add an S3 lifecycle rule as the
  backstop: DynamoDB reaping an expired pointer item does not delete its S3
  object.

## Clean up

```bash
aws dynamodb delete-table --table-name agent-storage
```
