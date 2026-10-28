// Reusable decoded-RGB example. Run via real_pipeline.py for verified offline photos.
import assert from 'node:assert/strict';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { invoke } from '../tools/moonvision_client.mjs';

const SIZE = 96;
const SEED = 20261028;
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];
const PIPELINE = [
  { op: 'letterbox', width: SIZE, height: SIZE, fill: [114, 114, 114], mask_value: 255, scale_up: false },
  { op: 'hflip', p: 0.5 },
];

async function checked(mode, request) {
  const response = await invoke(mode, request);
  assert.equal(response.ok, true, response.error);
  return response;
}

function annotatedItem(entry) {
  const { width, height } = entry.image;
  const left = Math.floor(width / 4), top = Math.floor(height / 4);
  const right = Math.ceil(width * 3 / 4), bottom = Math.ceil(height * 3 / 4);
  return {
    sample_id: entry.sample_id,
    sample: {
      image: entry.image,
      mask: Array.from({ length: width * height }, (_, p) =>
        p % width >= left && p % width < right && Math.floor(p / width) >= top && Math.floor(p / width) < bottom ? 1 : 0),
      boxes: [{ x_min: left, y_min: top, x_max: right, y_max: bottom, label: 'geometric-probe:rectangle' }],
    },
    keypoints: [{ x: width / 4, y: height / 3, label: 'geometric-probe:point', visible: true }],
  };
}

function checkGeometry(input, output) {
  const { width, height } = input.sample.image;
  const scale = Math.min(SIZE / width, SIZE / height, 1);
  const w = Math.min(SIZE, Math.max(1, Math.floor(width * scale + 0.5)));
  const h = Math.min(SIZE, Math.max(1, Math.floor(height * scale + 0.5)));
  const left = Math.floor((SIZE - w) / 2), top = Math.floor((SIZE - h) / 2);
  const flipped = output.replay.operations.some(op => op.op === 'hflip');
  assert.deepEqual(output.replay.operations, [
    { op: 'resize', width: w, height: h },
    { op: 'pad', left, top, right: SIZE - w - left, bottom: SIZE - h - top, fill: [114, 114, 114], mask_value: 255 },
    ...(flipped ? [{ op: 'hflip' }] : []),
  ]);
  const source = input.sample.boxes[0];
  const x0 = source.x_min * (w / width) + left, x1 = source.x_max * (w / width) + left;
  const expected = {
    x_min: flipped ? SIZE - x1 : x0, x_max: flipped ? SIZE - x0 : x1,
    y_min: source.y_min * (h / height) + top, y_max: source.y_max * (h / height) + top,
    label: source.label,
  };
  assert.deepEqual(output.sample.boxes, [expected]);
  const point = input.keypoints[0], x = point.x * (w / width) + left;
  assert.deepEqual(output.keypoints, [{ ...point, x: flipped ? SIZE - x : x, y: point.y * (h / height) + top }]);
  for (let y = 0; y < SIZE; y++) {
    for (let x = 0; x < SIZE; x++) {
      const dx = (flipped ? SIZE - 1 - x : x) - left, dy = y - top;
      const label = dx < 0 || dx >= w || dy < 0 || dy >= h ? 255 :
        input.sample.mask[Math.floor((2 * dy + 1) * height / (2 * h)) * width + Math.floor((2 * dx + 1) * width / (2 * w))];
      assert.equal(output.sample.mask[y * SIZE + x], label);
    }
  }
}

export async function runRealPipeline(entries) {
  assert.ok(Array.isArray(entries) && entries.length > 0 && entries.length * 3 * SIZE * SIZE <= 262144, 'batch exceeds JSON tensor budget');
  assert.equal(new Set(entries.map(entry => entry.sample_id)).size, entries.length, 'sample IDs must be unique');
  let total = 0;
  for (const entry of entries) {
    const { width, height, channels, data } = entry.image;
    assert.ok(Number.isInteger(width) && width > 0 && Number.isInteger(height) && height > 0);
    assert.equal(channels, 3, 'this example requires RGB photos');
    total += width * height * channels;
    assert.ok(total <= 1048576, 'decoded images exceed JSON budget');
    assert.equal(data.length, width * height * channels);
  }
  const items = entries.map(annotatedItem);
  const augmented = await checked('augment', { action: 'augment_batch', items, pipeline: PIPELINE, seed: SEED });
  augmented.items.forEach((output, i) => checkGeometry(items[i], output));
  const records = augmented.items.map(({ sample_id, replay }) => ({ sample_id, replay }));
  const repeated = await checked('augment', { action: 'replay_batch', items: [...items].reverse(), records });
  const byId = new Map(augmented.items.map(item => [item.sample_id, item]));
  for (const result of repeated.items) assert.deepEqual(result, byId.get(result.sample_id), 'ID-keyed replay must preserve pixels, masks and annotations');
  const images = augmented.items.map(item => item.sample.image);
  const plain = (await checked('interop', { action: 'pack', images })).tensor;
  const normalized = (await checked('interop', { action: 'pack', images, mean: MEAN, std: STD })).tensor;
  assert.deepEqual(plain.shape, [entries.length, 3, SIZE, SIZE]);
  assert.deepEqual(normalized.shape, plain.shape);
  let maxError = 0;
  for (let n = 0; n < images.length; n++) {
    for (let c = 0; c < 3; c++) {
      for (let p = 0; p < SIZE * SIZE; p++) {
        const index = (n * 3 + c) * SIZE * SIZE + p;
        const value = images[n].data[p * 3 + c] / 255;
        assert.equal(plain.data[index], value, 'NCHW channel/batch order');
        const error = Math.abs(normalized.data[index] - (value - MEAN[c]) / STD[c]);
        assert.ok(Number.isFinite(error) && error <= 1e-12, 'per-channel normalization');
        maxError = Math.max(maxError, error);
      }
    }
  }
  const restored = await checked('interop', { action: 'unpack', tensor: normalized, mean: MEAN, std: STD });
  assert.deepEqual(restored.images, images, 'denormalized bytes must round-trip exactly');
  return {
    model_input: normalized,
    previews: augmented.items,
    replays: {
      version: 1, seed: SEED, pipeline: PIPELINE, mean: MEAN, std: STD,
      annotation_policy: 'Synthetic geometric probes, not semantic ground truth: central integer-quarter rectangle mask (1 inside, 0 outside, 255 padding), matching box, and point at (width/4,height/3).',
      identity_policy: 'Verify source file and decoded-pixel hashes before reusing a record. Core Replay validates image shape, not identity.',
      records: records.map((record, i) => ({ ...record, source: entries[i].source })),
    },
    checks: {
      geometry_and_mask_formula: true, replay_by_id_byte_exact: true,
      nchw_index_formula: true, normalization_max_abs_error: maxError, normalized_byte_roundtrip: true,
      tensor_values: normalized.data.length,
    },
  };
}

if (process.argv[1] && fileURLToPath(import.meta.url) === path.resolve(process.argv[1])) {
  try {
    process.stdin.setEncoding('utf8');
    const chunks = [];
    let length = 0;
    for await (const chunk of process.stdin) {
      length += chunk.length;
      assert.ok(length <= 16000000, 'request exceeds transport budget');
      chunks.push(chunk);
    }
    const input = chunks.join('');
    console.log(JSON.stringify(await runRealPipeline(JSON.parse(input))));
  } catch (error) {
    console.error(`Real pipeline failed: ${error.message}`);
    process.exitCode = 1;
  }
}
