#!/usr/bin/env node
import fs from 'node:fs';
import readline from 'node:readline';
import { augment_json, warp_json, interop_json } from '../web/moonvision.js';

const [, , mode, filename] = process.argv;
const processJson = mode === 'augment' ? augment_json : mode === 'warp' ? warp_json : mode === 'interop' ? interop_json : null;
if (!processJson) {
  console.error('Usage: node tools/moonvision.mjs augment|warp|interop [request.json]\nWithout a file, reads one JSON request per stdin line.');
  process.exit(2);
}
if (filename) {
  const result = processJson(fs.readFileSync(filename, 'utf8'));
  console.log(result);
  if (!JSON.parse(result).ok) process.exitCode = 1;
} else {
  const lines = readline.createInterface({ input: process.stdin, crlfDelay: Infinity });
  for await (const line of lines) {
    if (line.trim()) console.log(processJson(line));
  }
}
