#!/usr/bin/env node
import { fileURLToPath } from 'url';

export function coreUtility(input: string): string {
  return `Processed: ${input}`;
}

const isDirectRun = process.argv[1] === fileURLToPath(import.meta.url);

if (isDirectRun) {
  const args = process.argv.slice(2);
  const inputArgument = args[0] || 'default-fallback-value';
  const output = coreUtility(inputArgument);
  console.log(output);
}
