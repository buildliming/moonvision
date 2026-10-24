const $ = (id) => document.getElementById(id);
const worker = new Worker(new URL('./worker.mjs', import.meta.url), { type: 'module' });
const pending = new Map();
let sequence = 0;
let sourceSample;
let augResult;
let lastReplay;
let warpSource;
let warpTarget;
let targetTheta;
let targetVersion = 0;
let busy = false;

function status(message, error = false) {
  $('status').textContent = message;
  $('status').classList.toggle('error', error);
}
function call(mode, request) {
  const id = ++sequence;
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject });
    worker.postMessage({ id, mode, request });
  });
}
worker.onmessage = async ({ data }) => {
  if (data.ready) {
    $('augment').disabled = false;
    $('register').disabled = false;
    try {
      await updateTarget();
      await augment();
      document.body.dataset.ready = 'true';
    } catch (error) { status(error.message, true); }
    return;
  }
  const task = pending.get(data.id);
  if (!task) return;
  pending.delete(data.id);
  if (data.result.ok) task.resolve(data);
  else task.reject(new Error(data.result.error || '计算失败'));
};
worker.onerror = (error) => {
  status('计算模块加载失败。请先运行 python tools/build_web.py，再通过本地服务器打开页面。', true);
  for (const task of pending.values()) task.reject(new Error(error.message || 'Worker failed'));
  pending.clear();
};
async function runBusy(message, action) {
  if (busy) return;
  busy = true;
  status(message);
  for (const id of ['augment', 'replay', 'register', 'reset', 'upload', 'replay-file', 'angle', 'tx', 'ty']) $(id).disabled = true;
  try { await action(); }
  catch (error) { status(error.message, true); }
  finally {
    busy = false;
    for (const id of ['augment', 'register', 'reset', 'upload', 'replay-file', 'angle', 'tx', 'ty']) $(id).disabled = false;
    $('replay').disabled = !lastReplay;
  }
}

function syntheticSample() {
  const width = 256, height = 192;
  const data = [], mask = [];
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const rect = x >= 28 && x < 112 && y >= 48 && y < 154;
      const circle = (x - 181) ** 2 + (y - 94) ** 2 < 38 ** 2;
      const label = rect ? 1 : circle ? 2 : 0;
      mask.push(label);
      if (rect) data.push(206 + Math.floor(x / 20), 116 + Math.floor(y / 15), 74, 255);
      else if (circle) data.push(48, 107 + Math.floor(x / 12), 108 + Math.floor(y / 14), 255);
      else {
        const stripe = Math.floor((x + y) / 24) % 2;
        data.push(231 + stripe * 7, 225 + stripe * 7, 207 + stripe * 7, 255);
      }
    }
  }
  return {
    image: { width, height, channels: 4, data }, mask,
    boxes: [
      { x_min: 28, y_min: 48, x_max: 112, y_max: 154, label: 'rectangle' },
      { x_min: 143, y_min: 56, x_max: 219, y_max: 132, label: 'circle' },
    ],
  };
}
const palette = [[0, 0, 0], [245, 176, 66], [80, 219, 196], [191, 142, 245]];
function drawSample(canvas, sample, overlays = true) {
  const { width, height, channels, data } = sample.image;
  canvas.width = width; canvas.height = height;
  const context = canvas.getContext('2d');
  const pixels = context.createImageData(width, height);
  for (let i = 0; i < width * height; i++) {
    for (let c = 0; c < 3; c++) pixels.data[i * 4 + c] = data[i * channels + (channels === 1 ? 0 : c)];
    pixels.data[i * 4 + 3] = channels === 4 ? data[i * channels + 3] : 255;
    const label = sample.mask?.[i] || 0;
    if (overlays && $('show-mask').checked && label) {
      const color = palette[1 + ((Math.abs(label) - 1) % 3)];
      for (let c = 0; c < 3; c++) pixels.data[i * 4 + c] = pixels.data[i * 4 + c] * .72 + color[c] * .28;
    }
  }
  context.putImageData(pixels, 0, 0);
  if (!overlays) return;
  context.lineWidth = Math.max(1.5, width / 160);
  context.font = `${Math.max(9, width / 24)}px monospace`;
  for (const [i, box] of sample.boxes.entries()) {
    const color = i % 2 ? '#72f1dc' : '#ffe0a0';
    context.strokeStyle = color;
    context.strokeRect(box.x_min, box.y_min, box.x_max - box.x_min, box.y_max - box.y_min);
    const textWidth = context.measureText(box.label).width;
    const textHeight = Math.max(13, width / 18);
    const textY = Math.max(textHeight, box.y_min);
    context.fillStyle = '#193e35';
    context.fillRect(box.x_min, textY - textHeight, textWidth + 7, textHeight);
    context.fillStyle = color;
    context.fillText(box.label, box.x_min + 3, textY - 3);
  }
}
function redrawAug() {
  if (sourceSample) drawSample($('source'), sourceSample);
  if (augResult) drawSample($('result'), augResult);
}
function setSource(sample) {
  sourceSample = sample;
  augResult = null; lastReplay = null;
  $('boxes').value = JSON.stringify(sample.boxes, null, 2);
  $('source-meta').textContent = `${sample.image.width} × ${sample.image.height}`;
  $('result-meta').textContent = '等待执行';
  $('replay').disabled = true; $('download').disabled = true;
  $('box-count').textContent = '—'; $('replay-result').textContent = '待验证'; $('aug-time').textContent = '—';
  $('replay-json').textContent = '执行增强后，可导出确定的变换参数。';
  drawSample($('source'), sample);
  const resultCanvas = $('result'); resultCanvas.width = sample.image.width; resultCanvas.height = sample.image.height;
}
function currentSample() {
  const boxes = JSON.parse($('boxes').value);
  if (!Array.isArray(boxes)) throw new Error('检测框必须为 JSON 数组。');
  return { ...sourceSample, boxes };
}
function pipeline() {
  const operations = [];
  if ($('crop').checked) operations.push({ op: 'random_crop', width: Math.max(1, Math.floor(sourceSample.image.width * .8)), height: Math.max(1, Math.floor(sourceSample.image.height * .8)) });
  if ($('flip').checked) operations.push({ op: 'hflip', p: .5 });
  if ($('rotate').checked) operations.push({ op: 'rotate90', turns: 1, p: .5 });
  operations.push({ op: 'resize', width: 192, height: 192 });
  if ($('color').checked) operations.push({ op: 'brightness', factor: 1.15, p: .7 }, { op: 'contrast', factor: 1.1, p: .7 });
  return operations;
}
async function augment() {
  await runBusy('正在同步变换图像、检测框与掩码…', async () => {
    const sample = currentSample();
    const seed = Number($('seed').value);
    if (!Number.isInteger(seed) || seed < 0 || seed > 2147483647) throw new Error('随机种子须为 0 到 2147483647 的整数。');
    const { result, elapsed } = await call('augment', { action: 'augment', sample, pipeline: pipeline(), seed, sample_id: $('sample-id').value, min_visibility: Number($('visibility').value) });
    augResult = result.sample; lastReplay = result.replay;
    sourceSample = sample;
    redrawAug();
    $('result-meta').textContent = `${augResult.image.width} × ${augResult.image.height}`;
    $('box-count').textContent = `${augResult.boxes.length} / ${sample.boxes.length}`;
    $('replay-result').textContent = '待验证';
    $('aug-time').textContent = `${elapsed.toFixed(1)} ms`;
    $('replay-json').textContent = JSON.stringify(lastReplay, null, 2);
    $('download').disabled = false;
    status('增强完成。可以更换种子探索结果，或验证同一记录的精确重放。');
  });
}
$('augment').onclick = augment;
$('replay').onclick = () => runBusy('正在使用记录重放…', async () => {
  const { result } = await call('augment', { action: 'replay', sample: currentSample(), replay: lastReplay });
  const same = JSON.stringify(result.sample) === JSON.stringify(augResult);
  $('replay-result').textContent = same ? '完全一致 ✓' : '输入已变化';
  status(same ? '重放验证通过：像素、掩码、检测框与标签全部一致。' : '当前输入与上次增强不同；请恢复原输入后重试。', !same);
});
$('download').onclick = () => {
  const blob = new Blob([JSON.stringify(lastReplay, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob), link = document.createElement('a');
  link.href = url; link.download = 'moonvision-replay.json'; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
};
$('replay-file').onchange = (event) => runBusy('正在读取重放记录…', async () => {
  const file = event.target.files[0]; if (!file) return;
  if (file.size > 2_000_000) throw new Error('重放记录超过 2 MB。');
  const replay = JSON.parse(await file.text());
  const { result } = await call('augment', { action: 'replay', sample: currentSample(), replay });
  augResult = result.sample; lastReplay = replay;
  drawSample($('result'), augResult);
  $('result-meta').textContent = `${augResult.image.width} × ${augResult.image.height}`;
  $('box-count').textContent = `${augResult.boxes.length} / ${currentSample().boxes.length}`;
  $('replay-result').textContent = '已导入'; $('download').disabled = false;
  $('replay-json').textContent = JSON.stringify(replay, null, 2);
  status('记录导入成功，已应用于当前样本。');
  event.target.value = '';
});
$('upload').onchange = (event) => runBusy('正在读取本地图片…', async () => {
  const file = event.target.files[0]; if (!file) return;
  const bitmap = await createImageBitmap(file);
  const scale = Math.min(1, 256 / Math.max(bitmap.width, bitmap.height));
  const canvas = document.createElement('canvas');
  canvas.width = Math.max(1, Math.round(bitmap.width * scale)); canvas.height = Math.max(1, Math.round(bitmap.height * scale));
  const context = canvas.getContext('2d'); context.drawImage(bitmap, 0, 0, canvas.width, canvas.height); bitmap.close();
  setSource({ image: { width: canvas.width, height: canvas.height, channels: 4, data: Array.from(context.getImageData(0, 0, canvas.width, canvas.height).data) }, mask: null, boxes: [] });
  status('图片已在本地加载。可以编辑检测框，然后执行增强。');
  event.target.value = '';
});
$('reset').onclick = () => { setSource(syntheticSample()); augment(); };
$('show-mask').onchange = redrawAug;
$('visibility').oninput = () => { $('visibility-value').textContent = Number($('visibility').value).toFixed(2); };

function makeWarpSource() {
  const size = 48, data = [];
  for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) {
    const u = (x + .5) / size * 2 - 1, v = (y + .5) / size * 2 - 1;
    data.push(.9 * Math.exp(-((u + .3) ** 2 / .045 + (v + .22) ** 2 / .12)) + .7 * Math.exp(-((u - .3) ** 2 / .08 + (v - .25) ** 2 / .04)) + .45 * Math.exp(-((u - .33) ** 2 / .025 + (v + .42) ** 2 / .025)));
  }
  return { shape: [1, 1, size, size], data };
}
function drawTensor(canvas, tensor) {
  const [, , height, width] = tensor.shape;
  const data = [];
  for (const value of tensor.data) {
    const t = Math.max(0, Math.min(1, value));
    data.push(Math.round(18 + t * 220), Math.round(43 + t * 173), Math.round(41 + t * 109), 255);
  }
  drawSample(canvas, { image: { width, height, channels: 4, data }, mask: null, boxes: [] }, false);
}
async function updateTarget() {
  const version = ++targetVersion;
  const angle = Number($('angle').value) * Math.PI / 180;
  const theta = [Math.cos(angle), -Math.sin(angle), Number($('tx').value), Math.sin(angle), Math.cos(angle), Number($('ty').value)];
  const { result } = await call('warp', { action: 'sample', input: warpSource, theta, out_height: 48, out_width: 48, align_corners: false });
  if (version !== targetVersion) return;
  targetTheta = theta; warpTarget = result.output;
  drawTensor($('warp-target'), warpTarget); drawTensor($('warp-result'), warpSource);
  for (const id of ['initial-loss', 'final-loss', 'improvement']) $(id).textContent = '—';
  $('loss-description').textContent = '目标已更新，开始配准后显示损失曲线。';
  $('loss-chart').getContext('2d').clearRect(0, 0, 800, 220);
}
for (const id of ['angle', 'tx', 'ty']) {
  $(id).oninput = () => {
    $(`${id}-value`).textContent = id === 'angle' ? `${$(id).value}°` : Number($(id).value).toFixed(2);
    updateTarget().catch((error) => status(error.message, true));
  };
}
function drawLosses(losses) {
  const canvas = $('loss-chart'), ctx = canvas.getContext('2d');
  const width = canvas.width, height = canvas.height, left = 60, top = 20, bottom = 34;
  ctx.clearRect(0, 0, width, height);
  const maximum = Math.max(...losses, 1e-12) * 1.05;
  ctx.font = '11px monospace'; ctx.lineWidth = 1;
  for (let i = 0; i < 5; i++) {
    const y = top + (height - top - bottom) * i / 4;
    ctx.strokeStyle = '#e1e8dd'; ctx.beginPath(); ctx.moveTo(left, y); ctx.lineTo(width - 20, y); ctx.stroke();
    ctx.fillStyle = '#657773'; ctx.fillText((maximum * (1 - i / 4)).toFixed(4), 4, y + 4);
  }
  ctx.strokeStyle = '#166c55'; ctx.lineWidth = 2.5; ctx.beginPath();
  losses.forEach((loss, i) => {
    const x = left + (width - left - 20) * i / Math.max(1, losses.length - 1);
    const y = top + (height - top - bottom) * (1 - loss / maximum);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke(); ctx.fillStyle = '#657773'; ctx.fillText('0', left, height - 10); ctx.fillText(`${losses.length - 1} iterations`, width - 125, height - 10);
}
$('register').onclick = () => runBusy('正在通过解析梯度优化仿射参数…', async () => {
  await updateTarget();
  const { result, elapsed } = await call('warp', { action: 'register', input: warpSource, target: warpTarget, iterations: Number($('iterations').value), learning_rate: .5, align_corners: false });
  drawTensor($('warp-result'), result.output);
  $('initial-loss').textContent = result.initial_loss.toExponential(3);
  $('final-loss').textContent = result.final_loss.toExponential(3);
  const reduction = result.initial_loss > 0 ? (1 - result.final_loss / result.initial_loss) * 100 : 0;
  $('improvement').textContent = `${reduction.toFixed(2)}%`;
  drawLosses(result.losses);
  $('theta-json').textContent = JSON.stringify({ target: targetTheta, learned: result.theta }, null, 2);
  $('loss-description').textContent = `${result.losses.length - 1} 次优化后，MSE 从 ${result.initial_loss.toExponential(4)} 降至 ${result.final_loss.toExponential(4)}。`;
  document.body.dataset.registration = 'complete';
  status(`配准完成，用时 ${elapsed.toFixed(1)} ms。参数、图像与每步损失均来自 MoonWarp。`);
});
for (const [tab, panel] of [['tab-aug', 'panel-aug'], ['tab-warp', 'panel-warp']]) {
  $(tab).onclick = () => {
    for (const [other, otherPanel] of [['tab-aug', 'panel-aug'], ['tab-warp', 'panel-warp']]) {
      $(other).setAttribute('aria-selected', String(other === tab)); $(other).tabIndex = other === tab ? 0 : -1; $(otherPanel).hidden = otherPanel !== panel;
    }
  };
  $(tab).onkeydown = (event) => {
    if (['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) {
      event.preventDefault(); const next = event.key === 'Home' ? 'tab-aug' : event.key === 'End' ? 'tab-warp' : tab === 'tab-aug' ? 'tab-warp' : 'tab-aug';
      $(next).click(); $(next).focus();
    }
  };
}
setSource(syntheticSample());
warpSource = makeWarpSource(); drawTensor($('warp-source'), warpSource); drawTensor($('warp-result'), warpSource);
