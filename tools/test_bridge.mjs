// End-to-end checks exercise the compiled MoonBit JS exports against independent fixtures.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { augment_json, warp_json } from '../web/moonvision.js';

let checks = 0, maximumWarpError = 0;
function invoke(fn, request) {
  const response = JSON.parse(fn(JSON.stringify(request)));
  assert.equal(response.ok, true, response.error);
  return response;
}
function close(actual, expected, tolerance = 1e-9) {
  assert.equal(actual.length, expected.length);
  let error = 0;
  for (let i = 0; i < actual.length; i++) {
    assert.ok(Number.isFinite(actual[i]), `nonfinite output at ${i}`);
    error = Math.max(error, Math.abs(actual[i] - expected[i]));
  }
  assert.ok(error <= tolerance, `max absolute error ${error} exceeds ${tolerance}`);
  return error;
}
function toSample(value) {
  return {
    image: { width: value.width, height: value.height, channels: value.channels, data: value.data },
    mask: value.mask,
    boxes: value.boxes.map(([x_min, y_min, x_max, y_max, label]) => ({ x_min, y_min, x_max, y_max, label })),
  };
}
function operation([name, ...values]) {
  switch (name) {
    case 'HFlip': return { op: 'hflip' };
    case 'VFlip': return { op: 'vflip' };
    case 'Rotate90': return { op: 'rotate90', turns: values[0] };
    case 'Resize': return { op: 'resize', width: values[0], height: values[1] };
    case 'Crop': return { op: 'crop', x: values[0], y: values[1], width: values[2], height: values[3] };
    case 'Brightness': return { op: 'brightness', factor: values[0] };
    case 'Contrast': return { op: 'contrast', factor: values[0] };
    default: throw new Error(`Unknown reference operation ${name}`);
  }
}
const augmentation = JSON.parse(fs.readFileSync(new URL('../verification/augmentation_reference.json', import.meta.url)));
for (const test of augmentation.cases) {
  const sample = toSample(test.input), expected = toSample(test.expected);
  const request = { action: 'augment', sample, pipeline: test.operations.map(operation), min_visibility: test.min_visibility, seed: 42, sample_id: test.name };
  const original = JSON.stringify(request);
  const result = invoke(augment_json, request);
  assert.equal(JSON.stringify(request), original, 'host request must not be changed');
  assert.deepEqual([result.sample.image.width, result.sample.image.height, result.sample.image.channels], [expected.image.width, expected.image.height, expected.image.channels]);
  close(result.sample.image.data, expected.image.data, 1);
  assert.deepEqual(result.sample.mask, expected.mask);
  assert.equal(result.sample.boxes.length, expected.boxes.length);
  for (let i = 0; i < expected.boxes.length; i++) {
    const actualBox = result.sample.boxes[i], expectedBox = expected.boxes[i];
    close(['x_min', 'y_min', 'x_max', 'y_max'].map(k => actualBox[k]), ['x_min', 'y_min', 'x_max', 'y_max'].map(k => expectedBox[k]));
    assert.equal(actualBox.label, expectedBox.label);
  }
  const replayed = invoke(augment_json, { action: 'replay', sample, replay: JSON.parse(JSON.stringify(result.replay)) });
  assert.deepEqual(replayed.sample, result.sample);
  checks++;
}
const warp = JSON.parse(fs.readFileSync(new URL('../verification/warp_reference.json', import.meta.url)));
for (const test of warp.sample_cases) {
  const result = invoke(warp_json, { action: 'sample', input: { shape: test.input_shape, data: test.input }, grid: { shape: test.grid_shape, data: test.grid }, align_corners: test.align_corners });
  assert.deepEqual(result.output.shape, test.output_shape);
  maximumWarpError = Math.max(maximumWarpError, close(result.output.data, test.output));
  checks++;
}
for (const test of warp.affine_cases) {
  const input = { shape: test.input_shape, data: test.input };
  const result = invoke(warp_json, { action: 'sample', input, theta: test.theta, out_height: test.output_shape[2], out_width: test.output_shape[3], align_corners: test.align_corners });
  maximumWarpError = Math.max(maximumWarpError, close(result.grid.data, test.grid), close(result.output.data, test.output));
  const grad = invoke(warp_json, { action: 'backward', input, theta: test.theta, grad_output: { shape: test.output_shape, data: test.grad_output }, align_corners: test.align_corners });
  maximumWarpError = Math.max(maximumWarpError, close(grad.input_grad.data, test.input_grad), close(grad.theta_grad, test.theta_grad));
  checks++;
}
// Malformed host input must remain a structured error, rather than escaping or truncating dimensions.
const invalidWarp = [
  '{', 'null', '{}',
  JSON.stringify({ action: 'sample', input: { shape: [1, 1, 1.5, 1], data: [1] }, theta: [1, 0, 0, 0, 1, 0], out_height: 1, out_width: 1 }),
  JSON.stringify({ action: 'sample', input: { shape: [1, 1, 1, 1], data: [1] }, theta: [1, 0, 0, 0, 1, 0], out_height: 1.5, out_width: 1 }),
];
for (const request of invalidWarp) {
  const result = JSON.parse(warp_json(request)); assert.equal(result.ok, false); assert.equal(typeof result.error, 'string'); checks++;
}
for (const request of ['{', 'null', '{}', '{"action":"unsupported"}']) {
  assert.equal(JSON.parse(augment_json(request)).ok, false); checks++;
}
console.log(JSON.stringify({ checks, maximum_warp_absolute_error: maximumWarpError, oracle: warp.metadata.torch, pixel_tolerance: 1, passed: true }, null, 2));
