# Transparent S3 offload for oversized values

Agent workloads produce oversized values routinely: a tool returns a whole
web page, a session accumulates a long transcript, a context offloader parks
a large intermediate result. DynamoDB items cap at 400 KB. This example makes
that cap invisible to the application.

## How it works

Configure the store with a bucket you own:

```python
storage = DynamoDBStorage(table, prefix="session/s1", s3_bucket="my-offload-bucket")
```

Values above the item-size limit are offloaded to Amazon S3, keyed by the
item's full storage key, with a small pointer item remaining in the table.
Values under the limit stay inline. `read` returns the full bytes either
way, `list` sees every key regardless of where its value lives, and deleting
a key reclaims the S3 object along with the item. Callers see one contract
regardless of payload size.

The script proves each of those statements: it writes 26 bytes and 1 MB
through the same store, reads both back intact, then peeks underneath with
raw DynamoDB and S3 calls to show the small value inline in its item, the
large one as an S3 object behind a pointer item, and the object gone after
delete.

Optional gzip compression (`compression="gzip"`) is applied before the
offload size check, so compressible values stay inline at lower cost.

## Run it

```bash
aws dynamodb create-table \
  --table-name agent-storage \
  --attribute-definitions AttributeName=pk,AttributeType=S AttributeName=sk,AttributeType=S \
  --key-schema AttributeName=pk,KeyType=HASH AttributeName=sk,KeyType=RANGE \
  --billing-mode PAY_PER_REQUEST

aws s3 mb s3://my-offload-bucket

pip install strands-dynamodb-storage boto3
python context_offload.py --table agent-storage --bucket my-offload-bucket
```

```text
wrote 26 B and 1,036,013 B through the same contract
read back 26 B and 1,036,013 B, both intact
list sees both: ['turns/t1', 'turns/t2-page']

turns/t1: DynamoDB item carries 26 B of data
turns/t2-page: DynamoDB item carries 0 B of data (pointer item)
S3 objects in bucket: [('session/s1/turns/t2-page', 1036013)]

after delete: S3 objects remaining: 0
```

## Where this matters in an agent

The Strands SDK's context offloader plugin parks oversized tool results
through the same `Storage` contract, so an agent constructed with this store
(`Agent(storage=storage, plugins=[ContextOffloader()])`) gets the S3
spillover for free: big tool outputs leave the model context and land wherever
the payload size says they should, with nothing in the agent code aware of
the split.

## Considerations

- If you enable TTL on offloaded values, add an S3 lifecycle rule as the
  backstop: DynamoDB reaping an expired pointer item does not delete its S3
  object.
- Runtime permissions extend to `s3:PutObject`, `s3:GetObject`, and
  `s3:DeleteObject` on the bucket prefix; the README in the repository root
  carries the full least-privilege policy.

## Clean up

```bash
aws dynamodb delete-table --table-name agent-storage
aws s3 rb s3://my-offload-bucket --force
```
