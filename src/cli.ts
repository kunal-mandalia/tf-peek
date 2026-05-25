#!/usr/bin/env node

// Patch before tfjs-node loads via dynamic import below — static imports are
// hoisted past synchronous code so this must live in a separate entry file.
const origEmitWarning = process.emitWarning.bind(process);
(process.emitWarning as Function) = (warning: string | Error, ...args: unknown[]) => {
  const code = typeof args[0] === 'string' ? args[1] : (args[0] as NodeJS.EmitWarningOptions)?.code;
  if (code === 'DEP0169') return;
  return origEmitWarning(warning, ...(args as [string]));
};

const { toStdOut } = await import('./peek.js');

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
