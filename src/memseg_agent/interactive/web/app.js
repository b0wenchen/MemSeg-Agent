'use strict';
const $ = id => document.getElementById(id);
const canvas = $('editor'), ctx = canvas.getContext('2d');
const overlay = document.createElement('canvas'), overlayCtx = overlay.getContext('2d');
let state = {memories: []}, image = null, imageURL = null, mask = null, caseId = null;
let mode = 'query', name = '', pendingMode = 'query', tool = 'brush';
let busy = false, dirty = false, reviewedMask = false, pointer = null, history = [], frame = null;

async function api(path, body) {
  const options = {headers: {'X-MemSeg-Request': '1'}};
  if (body !== undefined) {
    options.method = 'POST';
    options.headers['Content-Type'] = 'application/json';
    options.body = JSON.stringify(body);
  }
  const response = await fetch(`/api/${path}`, options);
  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.detail || `Request failed (${response.status})`);
  }
  return response.json();
}
function message(text, type = '') {
  $('status').textContent = text;
  $('status').className = type;
}
function controls() {
  document.querySelectorAll('button').forEach(button => { button.disabled = busy; });
  $('segment').disabled = busy || !image || mode !== 'query' || !state.memories.length;
  ['upload-mask', 'download-mask', 'brush-tool', 'erase-tool', 'clear-mask'].forEach(id => { $(id).disabled = busy || !image; });
  $('undo').disabled = busy || !image || !history.length;
  $('reviewed').disabled = busy || !image;
  $('save-mask').disabled = busy || !image || !$('reviewed').checked;
  $('mask-state').textContent = !image ? 'No mask' : reviewedMask ? 'Saved / reviewed mask' : dirty ? 'Edited mask · not saved' : 'Unreviewed mask';
}
async function run(text, action) {
  if (busy) return;
  busy = true; controls(); message(text, 'busy');
  try { await action(); } catch (error) { message(error.message || String(error), 'error'); }
  finally { busy = false; controls(); }
}
function loadImage(url) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error('Unable to decode the image'));
    img.src = url;
  });
}
function readFile(file) {
  if (!file || file.size > 20 * 1024 * 1024) return Promise.reject(new Error('Choose an image smaller than 20 MiB'));
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(new Error('Could not read the file'));
    reader.readAsDataURL(file);
  });
}
function setImage(img, url, title) {
  image = img; imageURL = url; name = title;
  canvas.width = overlay.width = image.naturalWidth;
  canvas.height = overlay.height = image.naturalHeight;
  mask = new Uint8Array(canvas.width * canvas.height);
  history = []; pointer = null; dirty = false; reviewedMask = false;
  $('reviewed').checked = false;
  $('empty-state').hidden = true; canvas.hidden = false;
  $('case-title').textContent = name;
  $('mode-label').textContent = mode === 'support' ? 'LABEL A FEW-SHOT EXAMPLE' : 'QUERY IMAGE';
  $('review-title').textContent = mode === 'support' ? 'Teach the target with this example' : 'Save an explicitly reviewed correction';
  $('review-text').textContent = mode === 'support' ? 'Paint the foreground or upload its binary PNG mask, review the overlay, then add it to few-shot memory.' : 'Correct the mask if necessary. Confirming saves this image and mask for later retrieval; it does not fine-tune the model.';
  $('save-mask').textContent = mode === 'support' ? 'Add to few-shot memory' : 'Save to working memory';
  $('dimensions').textContent = `${canvas.width} × ${canvas.height} px`;
  $('retrieval-list').replaceChildren();
  const p = document.createElement('p'); p.className = 'empty-list'; p.textContent = 'Run segmentation to see retrieved cases.';
  $('retrieval-list').append(p);
  resizeCanvas(); draw(); controls();
}
function resizeCanvas() {
  if (!image) return;
  const zoom = $('zoom').value;
  const scale = zoom === 'fit' ? Math.min(($('viewport').clientWidth - 34) / canvas.width, ($('viewport').clientHeight - 34) / canvas.height, 1) : Number(zoom);
  canvas.style.width = `${Math.max(1, canvas.width * scale)}px`;
  canvas.style.height = `${Math.max(1, canvas.height * scale)}px`;
}
function draw() {
  if (!image) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(image, 0, 0);
  const pixels = overlayCtx.createImageData(canvas.width, canvas.height);
  for (let i = 0; i < mask.length; i++) if (mask[i]) {
    pixels.data[i * 4] = 74; pixels.data[i * 4 + 1] = 231; pixels.data[i * 4 + 2] = 187; pixels.data[i * 4 + 3] = 255;
  }
  overlayCtx.putImageData(pixels, 0, 0);
  ctx.globalAlpha = Number($('opacity').value) / 100;
  ctx.drawImage(overlay, 0, 0); ctx.globalAlpha = 1;
}
function scheduleDraw() { if (frame === null) frame = requestAnimationFrame(() => { frame = null; draw(); }); }
function remember() { history.push(mask.slice()); if (history.length > 10) history.shift(); }
function edited() { dirty = true; reviewedMask = false; $('reviewed').checked = false; controls(); }
function point(event) {
  const box = canvas.getBoundingClientRect();
  return {x: (event.clientX - box.left) * canvas.width / box.width, y: (event.clientY - box.top) * canvas.height / box.height};
}
function dab(x, y) {
  const radius = Number($('brush-size').value) / 2;
  for (let yy = Math.max(0, Math.floor(y - radius)); yy < Math.min(canvas.height, Math.ceil(y + radius)); yy++) {
    for (let xx = Math.max(0, Math.floor(x - radius)); xx < Math.min(canvas.width, Math.ceil(x + radius)); xx++) {
      if ((xx + .5 - x) ** 2 + (yy + .5 - y) ** 2 <= radius ** 2) mask[yy * canvas.width + xx] = tool === 'brush' ? 1 : 0;
    }
  }
}
canvas.addEventListener('pointerdown', event => {
  if (!image || busy || event.button !== 0) return;
  event.preventDefault(); remember(); pointer = point(event);
  canvas.setPointerCapture(event.pointerId); dab(pointer.x, pointer.y); edited(); scheduleDraw();
});
canvas.addEventListener('pointermove', event => {
  if (!pointer || busy) return;
  const next = point(event), distance = Math.hypot(next.x - pointer.x, next.y - pointer.y);
  const steps = Math.max(1, Math.ceil(distance / Math.max(1, Number($('brush-size').value) / 4)));
  for (let i = 1; i <= steps; i++) dab(pointer.x + (next.x - pointer.x) * i / steps, pointer.y + (next.y - pointer.y) * i / steps);
  pointer = next; scheduleDraw();
});
['pointerup', 'pointercancel', 'lostpointercapture'].forEach(type => canvas.addEventListener(type, () => { pointer = null; }));
function selectTool(value) {
  tool = value;
  ['brush','erase'].forEach(key => { const button = $(`${key}-tool`); button.classList.toggle('selected', key === value); button.setAttribute('aria-pressed', String(key === value)); });
}
$('brush-tool').onclick = () => selectTool('brush');
$('erase-tool').onclick = () => selectTool('erase');
$('brush-size').oninput = () => { $('brush-label').textContent = `${$('brush-size').value} px`; };
$('opacity').oninput = scheduleDraw;
$('zoom').onchange = resizeCanvas;
window.addEventListener('resize', resizeCanvas);
$('undo').onclick = () => { if (history.length) { mask = history.pop(); edited(); draw(); } };
$('clear-mask').onclick = () => { if (confirm('Clear this mask? You can undo this change.')) { remember(); mask.fill(0); edited(); draw(); } };
$('reviewed').onchange = controls;
function chooseImage(nextMode) {
  if (dirty && !confirm('Discard unsaved edits and open another image?')) return;
  pendingMode = nextMode; $('image-file').value = ''; $('image-file').click();
}
$('add-example').onclick = $('empty-add').onclick = () => chooseImage('support');
$('open-query').onclick = () => chooseImage('query');
$('image-file').onchange = event => {
  const file = event.target.files[0]; if (!file) return;
  run('Opening image…', async () => {
    const url = await readFile(file);
    const result = await api(pendingMode === 'support' ? 'prepare' : 'query', {image: url, name: file.name});
    const loaded = await loadImage(result.image);
    mode = pendingMode; caseId = result.id || null;
    setImage(loaded, result.image, file.name);
    message(mode === 'support' ? 'Paint or upload the support mask, then review and save.' : 'Image ready. Segment it using the memory banks, or draw a mask manually.');
  });
};
async function maskFromURL(url) {
  const loaded = await loadImage(url);
  if (loaded.naturalWidth !== canvas.width || loaded.naturalHeight !== canvas.height) throw new Error('Mask dimensions must exactly match the image.');
  const temp = document.createElement('canvas'); temp.width = canvas.width; temp.height = canvas.height;
  const c = temp.getContext('2d'); c.drawImage(loaded, 0, 0);
  const pixels = c.getImageData(0, 0, temp.width, temp.height).data;
  const result = new Uint8Array(mask.length);
  for (let i = 0; i < result.length; i++) {
    const value = pixels[i * 4];
    if (![0,1,255].includes(value) || pixels[i * 4 + 1] !== value || pixels[i * 4 + 2] !== value || pixels[i * 4 + 3] !== 255) throw new Error('Use an opaque grayscale binary PNG (0/1 or 0/255), not a colored overlay.');
    result[i] = value > 0 ? 1 : 0;
  }
  return result;
}
function maskURL() {
  const temp = document.createElement('canvas'); temp.width = canvas.width; temp.height = canvas.height;
  const c = temp.getContext('2d'), pixels = c.createImageData(temp.width, temp.height);
  for (let i = 0; i < mask.length; i++) { pixels.data.fill(mask[i] * 255, i * 4, i * 4 + 3); pixels.data[i * 4 + 3] = 255; }
  c.putImageData(pixels, 0, 0); return temp.toDataURL('image/png');
}
$('upload-mask').onclick = () => { $('mask-file').value = ''; $('mask-file').click(); };
$('mask-file').onchange = event => {
  const file = event.target.files[0]; if (!file) return;
  run('Loading binary mask…', async () => {
    if (file.type !== 'image/png') throw new Error('Masks must be lossless PNG files.');
    const result = await maskFromURL(await readFile(file)); remember(); mask = result; edited(); draw(); message('Mask loaded. Inspect the overlay before saving.');
  });
};
function download(url, filename) { const anchor = document.createElement('a'); anchor.href = url; anchor.download = filename; document.body.append(anchor); anchor.click(); anchor.remove(); }
$('download-mask').onclick = () => download(maskURL(), `${name.replace(/\.[^.]+$/, '') || 'segmentation'}_mask.png`);
$('segment').onclick = () => {
  if (dirty && !confirm('Replace the current unsaved mask with a new prediction?')) return;
  run('Retrieving memories and running segmentation…', async () => {
    const result = await api('predict', {id: caseId});
    const nextMask = await maskFromURL(result.mask); remember(); mask = nextMask; dirty = false; reviewedMask = false; $('reviewed').checked = false; draw();
    $('retrieval-list').replaceChildren(...result.retrieved.map(item => {
      const card = document.createElement('div'); card.className = 'retrieved';
      const bank = document.createElement('span'); bank.className = 'bank'; bank.textContent = item.bank === 'fewshot' ? 'FEW-SHOT' : 'WORKING MEMORY';
      const title = document.createElement('strong'); title.textContent = item.name;
      const score = document.createElement('small'); score.textContent = `Cosine similarity: ${item.similarity.toFixed(3)}`;
      card.append(bank, title, score); return card;
    }));
    message('Prediction ready. Review it, correct errors, then explicitly save if you want to teach future cases.');
  });
};
async function refresh() {
  state = await api('state'); $('target-name').textContent = state.target;
  for (const [bank, listID, countID] of [['fewshot','support-list','support-count'],['working','working-list','working-count']]) {
    const cases = state.memories.filter(item => item.bank === bank); $(countID).textContent = cases.length;
    const rows = cases.map(item => {
      const row = document.createElement('div'); row.className = 'case-row';
      const open = document.createElement('button'); open.className = 'case-open';
      const title = document.createElement('span'); title.textContent = item.name;
      const details = document.createElement('small'); details.textContent = `${item.shape[1]}×${item.shape[0]} · ${item.foreground_pixels} foreground px`;
      open.append(title, details);
      open.onclick = () => run('Loading reviewed case…', async () => {
        const detail = await api(`memory/${item.id}`);
        $('memory-title').textContent = detail.name; $('memory-image').src = detail.image; $('memory-mask').src = detail.mask; $('memory-dialog').showModal(); message('Showing the stored image and reviewed mask.');
      });
      const remove = document.createElement('button'); remove.textContent = '×'; remove.className = 'case-remove'; remove.setAttribute('aria-label', `Remove ${item.name}`);
      remove.onclick = () => { if (confirm(`Remove ${item.name} from memory? Export first if you need to keep it.`)) run('Removing case…', async () => { await api('remove', {id: item.id}); await refresh(); message('Case removed from memory.'); }); };
      row.append(open, remove); return row;
    });
    if (!rows.length) { const empty = document.createElement('p'); empty.className = 'empty-list'; empty.textContent = bank === 'fewshot' ? 'Add your first labeled example.' : 'No confirmed corrections yet.'; rows.push(empty); }
    $(listID).replaceChildren(...rows);
  }
  $('capacity').textContent = `${state.memories.length} / ${state.capacity} reviewed cases · top-${state.top_k} per bank`;
  controls();
}
$('save-mask').onclick = () => run('Encoding the reviewed image and mask…', async () => {
  if (!$('reviewed').checked) throw new Error('Review and confirm the mask first.');
  if (mode === 'support') await api('support', {image: imageURL, mask: maskURL(), name});
  else await api('correct', {id: caseId, mask: maskURL(), confirmed: true});
  dirty = false; reviewedMask = true; $('reviewed').checked = false;
  await refresh(); message(mode === 'support' ? 'Example saved. Open a query image to start segmenting.' : 'Correction saved to working memory. It can now guide later queries.');
});
function clearEditor() {
  image = imageURL = mask = caseId = null; dirty = false; reviewedMask = false; history = []; pointer = null;
  canvas.hidden = true; $('empty-state').hidden = false; $('reviewed').checked = false;
  $('case-title').textContent = 'Begin with a labeled example'; $('dimensions').textContent = 'No image loaded'; $('retrieval-list').replaceChildren();
}
$('new-session').onclick = () => {
  if (!confirm('Clear all in-memory examples and edits? Export reviewed memories first if needed.')) return;
  const target = prompt('What single target will you segment?', state.target); if (target === null) return;
  run('Starting a new task…', async () => { await api('reset', {confirmed: true, target}); clearEditor(); await refresh(); message('New empty session. Add a labeled example.'); });
};
$('export-session').onclick = () => run('Preparing reviewed memories for download…', async () => {
  if (!confirm('This archive contains your images and masks. Download to a secure location?')) { message('Export cancelled.'); return; }
  const response = await fetch('/api/export', {headers: {'X-MemSeg-Request': '1'}});
  if (!response.ok) { const error = await response.json().catch(() => ({})); throw new Error(error.detail || 'Export failed'); }
  const url = URL.createObjectURL(await response.blob()); download(url, 'memseg-reviewed-memories.zip'); setTimeout(() => URL.revokeObjectURL(url), 30000);
  message('Memory archive exported. Treat it as sensitive data.');
});
$('import-session').onclick = () => { $('archive-file').value = ''; $('archive-file').click(); };
$('archive-file').onchange = event => {
  const file = event.target.files[0]; if (!file) return;
  run('Importing and re-encoding reviewed memories…', async () => {
    if (file.size > 192 * 1024 * 1024) throw new Error('Archive exceeds 192 MiB.');
    if (state.memories.length || image) throw new Error('Start a new empty session before importing.');
    const response = await fetch('/api/import', {method: 'POST', headers: {'X-MemSeg-Request': '1', 'Content-Type': 'application/zip'}, body: file});
    const result = await response.json(); if (!response.ok) throw new Error(result.detail || 'Import failed');
    await refresh(); message('Reviewed cases restored. Open a query image to continue.');
  });
};
$('close-dialog').onclick = () => $('memory-dialog').close();
window.addEventListener('beforeunload', event => { if (dirty) { event.preventDefault(); event.returnValue = ''; } });
run('Connecting to the local model…', async () => { await refresh(); message('Ready. Add a support example, or import a saved memory archive.'); });
