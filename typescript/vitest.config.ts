import { defineConfig } from 'vitest/config'

// Unit tests only — fast, offline, no AWS. Integration tests live under test/integ
// and run via `npm run test:integ` (vitest.integ.config.ts).
export default defineConfig({
  resolve: {
    // The source uses NodeNext-style `.js` import specifiers for `.ts` files.
    extensionAlias: {
      '.js': ['.ts', '.js'],
    },
  },
  test: {
    include: ['src/**/*.test.ts'],
    exclude: ['test/integ/**', 'node_modules/**', 'dist/**'],
    // Used by `npm run test:coverage`; CI uploads the lcov report to Codecov.
    coverage: {
      provider: 'v8',
      include: ['src/**/*.ts'],
      exclude: ['src/**/*.test.ts'],
      reporter: ['text', 'lcov'],
    },
  },
})
