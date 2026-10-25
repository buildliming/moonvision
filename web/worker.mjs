import { augment_json, warp_json, interop_json } from './moonvision.js';

self.onmessage = ({ data: { id, mode, request } }) => {
  const started = performance.now();
  try {
    const processJson = mode === 'augment' ? augment_json : mode === 'interop' ? interop_json : warp_json;
    const result = JSON.parse(processJson(JSON.stringify(request)));
    self.postMessage({ id, result, elapsed: performance.now() - started });
  } catch (error) {
    self.postMessage({ id, result: { ok: false, error: String(error.message || error) } });
  }
};
self.postMessage({ ready: true });
