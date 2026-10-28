#!/usr/bin/env node
import fs from 'node:fs';
import { processJson, requestLimits } from './moonvision_client.mjs';

function limitError(limit) {
  return JSON.stringify({ ok: false, error: `request exceeds ${limit} UTF-16 code units` });
}

// Await each write so a slow pipe consumer cannot accumulate an unbounded queue.
function writeLine(line) {
  return new Promise((resolve, reject) => {
    process.stdout.write(`${line}\n`, (error) => error ? reject(error) : resolve());
  });
}

// Bound each line before calling MoonBit. Discard an oversized line up to its
// delimiter, then continue with the next request in the same process.
async function* boundedLines(stream, limit) {
  stream.setEncoding('utf8');
  let parts = [], length = 0, oversized = false, afterCR = false;
  for await (const chunk of stream) {
    let start = 0;
    if (afterCR && chunk[0] === '\n') start = 1;
    afterCR = false;
    for (let i = start; i <= chunk.length; i++) {
      if (i < chunk.length && chunk[i] !== '\r' && chunk[i] !== '\n') continue;
      if (!oversized) {
        length += i - start;
        if (length > limit) { oversized = true; parts = []; }
        else if (i > start) parts.push(chunk.slice(start, i));
      }
      if (i === chunk.length) break;
      yield oversized ? { error: limitError(limit) } : { text: parts.join('') };
      parts = []; length = 0; oversized = false;
      if (chunk[i] === '\r') {
        if (chunk[i + 1] === '\n') i++;
        else if (i === chunk.length - 1) afterCR = true;
      }
      start = i + 1;
    }
  }
  if (oversized) yield { error: limitError(limit) };
  else if (length) yield { text: parts.join('') };
}

async function readRequest(filename, limit) {
  const parts = [];
  let length = 0;
  for await (const chunk of fs.createReadStream(filename, { encoding: 'utf8' })) {
    length += chunk.length;
    if (length > limit) return { error: limitError(limit) };
    parts.push(chunk);
  }
  return { text: parts.join('') };
}

async function main() {
  const [, , mode, filename, ...extra] = process.argv;
  if (!Object.hasOwn(requestLimits, mode) || extra.length) {
    console.error('Usage: node tools/moonvision.mjs augment|warp|interop [request.json]\nWithout a file, reads one JSON request per stdin line.');
    process.exitCode = 2;
    return;
  }
  const limit = requestLimits[mode];
  if (filename) {
    let request;
    try { request = await readRequest(filename, limit); }
    catch (error) {
      request = { error: JSON.stringify({ ok: false, error: `Cannot read request file: ${error.message}` }) };
    }
    const result = request.error ?? await processJson(mode, request.text);
    await writeLine(result);
    if (!JSON.parse(result).ok) process.exitCode = 1;
  } else {
    for await (const request of boundedLines(process.stdin, limit)) {
      if (request.error) await writeLine(request.error);
      else if (request.text.trim()) await writeLine(await processJson(mode, request.text));
    }
  }
}

// Write callbacks handle errors; keep the accompanying stream error event from
// becoming an uncaught exception when a consumer closes its pipe early.
process.stdout.on('error', () => {});
main().catch((error) => {
  if (error.code === 'EPIPE') return;
  console.error(`MoonVision CLI failed: ${error.message || String(error)}`);
  process.exitCode = 1;
});
