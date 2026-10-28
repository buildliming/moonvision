// Reusable Node/browser entry point for the compiled MoonBit JSON bridge.
// Importing this module does not load the runtime or start a CLI.
export const requestLimits = Object.freeze({ augment: 16_777_216, warp: 16_000_000, interop: 16_000_000 });
const exportNames = Object.freeze({ augment: 'augment_json', warp: 'warp_json', interop: 'interop_json' });
let runtime;

function failure(message) {
  return JSON.stringify({ ok: false, error: message });
}

/** Return a JSON response string. Limits count UTF-16 code units, like MoonBit. */
export async function processJson(mode, text) {
  if (typeof mode !== 'string' || !Object.hasOwn(requestLimits, mode)) {
    return failure('mode must be augment, warp, or interop');
  }
  if (typeof text !== 'string') return failure('request must be a JSON string');
  if (text.length > requestLimits[mode]) {
    return failure(`request exceeds ${requestLimits[mode]} UTF-16 code units`);
  }
  try {
    runtime ??= import('../web/moonvision.js');
    const bridge = await runtime;
    return bridge[exportNames[mode]](text);
  } catch (error) {
    if (error?.code === 'ERR_MODULE_NOT_FOUND') {
      return failure('Compiled MoonBit bridge is missing; run python tools/build_web.py first.');
    }
    return failure(`MoonBit bridge failed: ${error?.message || String(error)}`);
  }
}

/** Invoke a request object and return its parsed response, including ok:false errors. */
export async function invoke(mode, request) {
  try {
    return JSON.parse(await processJson(mode, JSON.stringify(request)));
  } catch (error) {
    return { ok: false, error: `Request serialization failed: ${error?.message || String(error)}` };
  }
}
