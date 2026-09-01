/**
 * Store values beyond the DynamoDB item limit with transparent S3 offload.
 *
 * Agent workloads produce oversized values routinely: a tool returns a whole
 * web page, a session accumulates a long transcript, a context offloader parks
 * a large intermediate result. DynamoDB items cap at 400 KB. Configuring the
 * store with an S3 bucket makes that cap invisible: values above the limit are
 * offloaded to S3 with a small pointer item remaining in the table, and
 * `read` returns the full bytes either way. Callers see one contract
 * regardless of payload size.
 *
 * The script writes a small value and a 1 MB value through the same store,
 * reads both back, then peeks underneath to show where each physically lives:
 * the small value inline in its DynamoDB item, the large one as an S3 object
 * behind a pointer item. Deleting the key reclaims the S3 object too.
 *
 * Prerequisites:
 *   - A pk/sk table (see README.md) and an S3 bucket you own.
 *   - Credentials for the table plus s3:PutObject/GetObject/DeleteObject on
 *     the bucket.
 *
 * Usage:
 *   npx tsx context-offload.ts --table agent-storage --bucket my-offload-bucket
 */

import { parseArgs } from 'node:util';
import assert from 'node:assert/strict';

import { DynamoDBClient, GetItemCommand } from '@aws-sdk/client-dynamodb';
import { ListObjectsV2Command, S3Client } from '@aws-sdk/client-s3';
import { DynamoDBStorage } from 'strands-dynamodb-storage';

const LARGE_SIZE = 1_000_000; // 1 MB, over the 400 KB item limit

async function main(table: string, bucket: string, region: string): Promise<void> {
  const storage = new DynamoDBStorage(table, { region, prefix: 'session/s1', s3Bucket: bucket });

  const small = Buffer.from('user asked about seat maps');
  const large = Buffer.concat([
    Buffer.from('<html>'),
    Buffer.from('tool result far too big for one item '.repeat(28_000)),
    Buffer.from('</html>'),
  ]);
  assert.ok(large.length > LARGE_SIZE);

  await storage.write('turns/t1', small);
  await storage.write('turns/t2-page', large);
  console.log(`wrote ${small.length} B and ${large.length.toLocaleString('en-US')} B through the same contract`);

  // Reads are symmetric: full bytes come back either way.
  const r1 = await storage.read('turns/t1');
  const r2 = await storage.read('turns/t2-page');
  assert.ok(r1 !== null && Buffer.from(r1).equals(small));
  assert.ok(r2 !== null && Buffer.from(r2).equals(large));
  console.log(`read back ${r1.length} B and ${r2.length.toLocaleString('en-US')} B, both intact`);
  console.log(`list sees both: ${JSON.stringify([...(await storage.list('turns/'))].sort())}\n`);

  // Peek underneath at where each value physically lives.
  const ddb = new DynamoDBClient({ region });
  const s3 = new S3Client({ region });
  for (const sk of ['turns/t1', 'turns/t2-page']) {
    const { Item } = await ddb.send(
      new GetItemCommand({ TableName: table, Key: { pk: { S: 'session/s1' }, sk: { S: sk } } }),
    );
    const size = Item?.data?.B?.length ?? 0;
    console.log(`${sk}: DynamoDB item carries ${size} B of data${size ? '' : ' (pointer item)'}`);
  }
  const objects = (await s3.send(new ListObjectsV2Command({ Bucket: bucket }))).Contents ?? [];
  console.log(`S3 objects in bucket: ${JSON.stringify(objects.map((o) => [o.Key, o.Size]))}\n`);

  // Delete reclaims the S3 object along with the item.
  await storage.delete('turns/t2-page');
  await storage.delete('turns/t1');
  const remaining = (await s3.send(new ListObjectsV2Command({ Bucket: bucket }))).KeyCount ?? 0;
  console.log(`after delete: S3 objects remaining: ${remaining}`);
}

const { values } = parseArgs({
  options: {
    table: { type: 'string', default: 'agent-storage' },
    bucket: { type: 'string' },
    region: { type: 'string', default: 'us-east-1' },
  },
});
if (values.bucket === undefined) {
  console.error('error: the following argument is required: --bucket');
  process.exit(2);
}

await main(values.table, values.bucket, values.region);
