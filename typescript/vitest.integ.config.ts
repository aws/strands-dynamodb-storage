import { defineConfig } from 'vitest/config'

// Integration tests — run against REAL DynamoDB + S3 (never DynamoDB Local).
// Gated by RUN_INTEG=1 and require AWS credentials in the environment.
// Table + bucket are provisioned and torn down by the suite itself.
export default defineConfig({
  resolve: {
    extensionAlias: {
      '.js': ['.ts', '.js'],
    },
  },
  test: {
    include: ['test/integ/**/*.test.ts'],
    testTimeout: 120_000,
    hookTimeout: 300_000,
    // Real network calls to one shared table/bucket — keep it serial.
    fileParallelism: false,
    sequence: { concurrent: false },
    retry: 0,
  },
})
