#!/usr/bin/env node
import { toStdOut } from './peek.js';

function parseArgs(argv: string[]): Record<string, string> {
  const result: Record<string, string> = {};
  for (const arg of argv) {
    const match = arg.match(/^--?([^=]+)=(.*)$/);
    if (match) result[match[1]] = match[2];
  }
  return result;
}

const { modelPath, values, format } = parseArgs(process.argv.slice(2));
const options: import('./peek.js').PeekOptions = {
  ...(values === 'false' && { values: false }),
  format: format === 'json' ? 'json' : 'markdown',
};
toStdOut(modelPath, options).catch((e: unknown) => {
  process.stderr.write(String(e) + '\n');
  process.exit(1);
});
