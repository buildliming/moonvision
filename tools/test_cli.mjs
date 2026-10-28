// Transport tests exercise the real executable and compiled MoonBit bridge.
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { once } from 'node:events';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { invoke, processJson, requestLimits } from './moonvision_client.mjs';

const root = fileURLToPath(new URL('..', import.meta.url));
const executable = path.join(root, 'tools/moonvision.mjs');
const request = { action: 'sample', input: { shape: [1, 1, 1, 1], data: [7] }, theta: [1, 0, 0, 0, 1, 0], out_height: 1, out_width: 1 };
const valid = JSON.stringify(request);
let checks = 0;

async function cli(args, chunks = [], { executablePath = executable, slow = false } = {}) {
  const child = spawn(process.execPath, [executablePath, ...args], { cwd: root, stdio: ['pipe', 'pipe', 'pipe'] });
  let stdout = '', stderr = '';
  child.stdout.setEncoding('utf8'); child.stderr.setEncoding('utf8');
  child.stdout.on('data', (chunk) => {
    stdout += chunk;
    if (slow) { child.stdout.pause(); setTimeout(() => child.stdout.resume(), 2); }
  });
  child.stderr.on('data', (chunk) => { stderr += chunk; });
  const timer = setTimeout(() => child.kill(), 30_000);
  const completion = once(child, 'close');
  try {
    for (const chunk of chunks) {
      if (!child.stdin.write(chunk)) await once(child.stdin, 'drain');
    }
    child.stdin.end();
    const [code, signal] = await completion;
    assert.equal(signal, null, 'CLI timed out');
    return { code, stdout, stderr, lines: stdout.trim() ? stdout.trim().split('\n').map(JSON.parse) : [] };
  } finally { clearTimeout(timer); }
}

for (const mode of ['invalid', '__proto__', 'toString', null]) {
  assert.equal((await invoke(mode, request)).ok, false);
  checks++;
}
assert.equal((await invoke('warp', request)).output.data[0], 7);
assert.equal(JSON.parse(await processJson('warp', null)).ok, false);
const cyclic = {}; cyclic.self = cyclic;
assert.equal((await invoke('warp', cyclic)).ok, false);
checks++;
const exactWarp = valid + ' '.repeat(requestLimits.warp - valid.length);
assert.equal(JSON.parse(await processJson('warp', exactWarp)).ok, true);
assert.equal(JSON.parse(await processJson('warp', exactWarp + ' ')).ok, false);
checks++;

const imported = await new Promise((resolve, reject) => {
  const child = spawn(process.execPath, ['--input-type=module', '-e', 'await import("./tools/moonvision_client.mjs"); console.log("imported")'], { cwd: root });
  let output = ''; child.stdout.on('data', chunk => { output += chunk; });
  child.on('error', reject); child.on('close', code => resolve({ code, output }));
});
assert.deepEqual(imported, { code: 0, output: 'imported\n' }); checks++;

let result = await cli(['warp'], ['\n{}\r', '\n', valid.slice(0, 25), valid.slice(25)]);
assert.equal(result.code, 0); assert.equal(result.stderr, '');
assert.equal(result.lines.length, 2); assert.equal(result.lines[0].ok, false);
assert.equal(result.lines[1].output.data[0], 7); checks++;

const temporary = await fs.mkdtemp(path.join(os.tmpdir(), 'moonvision-cli-'));
try {
  const filename = path.join(temporary, 'request.json');
  await fs.writeFile(filename, valid);
  result = await cli(['warp', filename]);
  assert.equal(result.code, 0); assert.equal(result.lines[0].ok, true); checks++;
  for (const target of [path.join(temporary, 'missing.json'), temporary]) {
    result = await cli(['warp', target]);
    assert.equal(result.code, 1); assert.equal(result.stderr, '');
    assert.equal(result.lines[0].ok, false); assert.match(result.lines[0].error, /Cannot read request file/); checks++;
  }
  await fs.writeFile(filename, '{}');
  result = await cli(['warp', filename]);
  assert.equal(result.code, 1); assert.equal(result.lines[0].ok, false); checks++;

  // These labels occupy more UTF-8 bytes than UTF-16 units. Exact-limit input
  // must be accepted, including when a multi-byte character crosses a chunk.
  const augment = JSON.stringify({ action: 'augment', sample: { image: { width: 1, height: 1, channels: 1, data: [7] }, boxes: [] }, pipeline: [], sample_id: '照片🌓' });
  const prefixLength = 65_535 - Buffer.byteLength(augment.slice(0, augment.indexOf('照片')));
  const exact = ' '.repeat(prefixLength) + augment + ' '.repeat(requestLimits.augment - augment.length - prefixLength);
  await fs.writeFile(filename, exact);
  result = await cli(['augment', filename]);
  assert.equal(result.code, 0); assert.equal(result.lines[0].ok, true); checks++;
  await fs.appendFile(filename, ' ');
  result = await cli(['augment', filename]);
  assert.equal(result.code, 1); assert.match(result.lines[0].error, /16777216 UTF-16/); checks++;

  // One oversize line yields exactly one error and does not poison later input.
  result = await cli(['warp'], ['x'.repeat(requestLimits.warp), 'x\r', '\n', valid]);
  assert.equal(result.code, 0); assert.equal(result.lines.length, 2);
  assert.match(result.lines[0].error, /16000000 UTF-16/);
  assert.equal(result.lines[1].ok, true); checks++;
  result = await cli(['warp'], ['x'.repeat(requestLimits.warp + 1)]);
  assert.equal(result.lines.length, 1); assert.equal(result.lines[0].ok, false); checks++;

  // Copy just the entry points: help and imports work without a compiled asset,
  // while requests explain the missing build through the usual JSON contract.
  await fs.mkdir(path.join(temporary, 'tools'));
  for (const name of ['moonvision.mjs', 'moonvision_client.mjs']) {
    await fs.copyFile(path.join(root, 'tools', name), path.join(temporary, 'tools', name));
  }
  const isolated = path.join(temporary, 'tools/moonvision.mjs');
  result = await cli(['warp'], [valid], { executablePath: isolated });
  assert.equal(result.code, 0); assert.match(result.lines[0].error, /python tools\/build_web.py/); checks++;
  result = await cli([], [], { executablePath: isolated });
  assert.equal(result.code, 2); assert.match(result.stderr, /Usage:/); checks++;

  // Larger replies and a deliberately paused reader exercise pipe backpressure.
  const large = JSON.stringify({ ...request, input: { shape: [1, 1, 1, 4096], data: Array(4096).fill(7) }, out_width: 4096 });
  result = await cli(['warp'], Array(40).fill(`${large}\n`), { slow: true });
  assert.equal(result.code, 0); assert.equal(result.lines.length, 40);
  for (const line of result.lines) { assert.equal(line.ok, true); assert.equal(line.output.data.length, 4096); }
  checks++;
} finally {
  assert.equal(path.dirname(path.resolve(temporary)), path.resolve(os.tmpdir()));
  assert.ok(path.basename(temporary).startsWith('moonvision-cli-'));
  await fs.rm(temporary, { recursive: true, force: true });
}

console.log(JSON.stringify({ checks, passed: true }));
