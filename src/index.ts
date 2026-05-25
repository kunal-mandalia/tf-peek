#!/usr/bin/env node
import { fileURLToPath } from 'url';
import { realpathSync } from 'fs';
import { toStdOut } from './peek.js';
export { toStdOut, peekLayers } from './peek.js';
export type { LayerData } from './peek.js';


export function coreUtility(input: string): string {
  return `Processed: ${input}`;
}

const isDirectRun = process.argv[1] != null &&
  realpathSync(process.argv[1]) === realpathSync(fileURLToPath(import.meta.url));

function parseArgs(argv: string[]): Record<string, string> {
  const result: Record<string, string> = {};
  for (const arg of argv) {
    const match = arg.match(/^--?([^=]+)=(.*)$/);
    if (match) result[match[1]] = match[2];
  }
  return result;
}

if (isDirectRun) {
  const { modelPath } = parseArgs(process.argv.slice(2));

  toStdOut(modelPath).catch((e: unknown) => {
    process.stderr.write(String(e) + '\n');
    process.exit(1);
  });
}
