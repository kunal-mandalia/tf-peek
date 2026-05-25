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

const { modelPath } = parseArgs(process.argv.slice(2));
toStdOut(modelPath).catch((e: unknown) => {
  process.stderr.write(String(e) + '\n');
  process.exit(1);
});
