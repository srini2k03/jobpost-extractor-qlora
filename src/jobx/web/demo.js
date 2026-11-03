// In-browser demo: the fine-tuned Qwen2.5-0.5B LoRA (GGUF Q4_K_M) running locally via wllama (llama.cpp WASM).
// Nothing is sent to a server; the model is downloaded once from this site and cached by the browser.
import { Wllama } from 'https://cdn.jsdelivr.net/npm/@wllama/wllama@__WLLAMA_VERSION__/esm/index.js';

const WASM = { default: 'https://cdn.jsdelivr.net/npm/@wllama/wllama@__WLLAMA_VERSION__/src/wasm/wllama.wasm' };
const $ = (id) => document.getElementById(id);
const load = async (p) => (await fetch(p)).json();

const [manifest, prompt, schema, aliases, examples] = await Promise.all(
  ['model/manifest.json', 'demo/prompt.json', 'demo/schema.json', 'demo/aliases.json', 'demo/examples.json'].map(load),
);
const KEYS = ['company', 'role', 'seniority', 'skills', 'min_experience_years', 'max_experience_years', 'salary',
  'locations', 'work_mode', 'employment_type', 'education'];
let wllama = null;

$('size-note').textContent = `${manifest.size_mb} MB, downloaded once and cached by your browser`;
examples.forEach((ex, i) => {
  const b = document.createElement('button');
  b.textContent = `Example ${i + 1}`;
  b.className = 'secondary';
  b.dataset.example = String(i);
  b.onclick = () => { $('posting').value = ex.text; };
  $('examples').appendChild(b);
});
$('posting').value = examples[0]?.text || '';

function setStatus(text, state) {
  $('status').textContent = text;
  $('status').dataset.state = state || '';
}

async function loadModel() {
  if (wllama) return;
  $('load').disabled = true;
  setStatus('Loading the model…', 'loading');
  const w = new Wllama(WASM, { parallelDownloads: 4 });
  const first = new URL(manifest.parts[0], location.href).href;
  await w.loadModelFromUrl(first, {
    n_ctx: 2048,
    progressCallback: ({ loaded, total }) => setStatus(`Downloading ${Math.round((100 * loaded) / total)}%`, 'loading'),
  });
  wllama = w;
  $('extract').disabled = false;
  setStatus('Model ready. Paste a posting or pick an example, then Extract.', 'ready');
}

function validate(obj) {
  if (typeof obj !== 'object' || obj === null || Array.isArray(obj)) return 'not a JSON object';
  const missing = KEYS.filter((k) => !(k in obj));
  return missing.length ? `missing keys: ${missing.join(', ')}` : null;
}

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
}

function highlight(text, out) {
  const terms = [];
  const add = (label, values) => values.filter(Boolean).forEach((v) => terms.push({ label, v: String(v) }));
  add('company', [out.company]);
  add('role', [out.role]);
  for (const s of out.skills || []) add('skill', [s, ...(aliases.skills[s] || [])]);
  for (const c of out.locations || []) add('location', [c, ...(aliases.locations[c] || [])]);
  for (const f of ['work_mode', 'employment_type', 'education']) {
    if (out[f]) add(f, aliases[f][out[f]] || []);
  }
  const spans = [];
  const lower = text.toLowerCase();
  for (const t of terms.sort((a, b) => b.v.length - a.v.length)) {
    const needle = t.v.toLowerCase();
    let i = lower.indexOf(needle);
    while (needle.length > 1 && i !== -1) {
      const end = i + needle.length;
      const word = !/[a-z0-9]/i.test(text[i - 1] || ' ') && !/[a-z0-9]/i.test(text[end] || ' ');
      if (word && !spans.some((s) => i < s.end && s.start < end)) spans.push({ start: i, end, label: t.label });
      i = lower.indexOf(needle, end);
    }
  }
  spans.sort((a, b) => a.start - b.start);
  let html = '';
  let pos = 0;
  for (const s of spans) {
    html += escapeHtml(text.slice(pos, s.start)) +
      `<mark class="f-${s.label}" title="${s.label}">${escapeHtml(text.slice(s.start, s.end))}</mark>`;
    pos = s.end;
  }
  return html + escapeHtml(text.slice(pos));
}

async function extract() {
  const text = $('posting').value.trim();
  if (!text) return;
  $('extract').disabled = true;
  setStatus('Extracting…', 'busy');
  const t0 = performance.now();
  const req = {
    messages: [
      { role: 'system', content: prompt.system },
      { role: 'user', content: prompt.user_template.replace('{text}', text) },
    ],
    max_tokens: 400,
    temperature: 0,
  };
  if ($('guided').checked) req.response_format = { type: 'json_schema', json_schema: { name: 'JobPosting', schema } };
  try {
    const res = await wllama.createChatCompletion(req);
    const raw = res.choices[0].message.content || '';
    const m = raw.match(/\{[\s\S]*\}/);
    let obj = null;
    let err = 'no JSON object in the output';
    if (m) {
      try { obj = JSON.parse(m[0]); err = validate(obj); } catch (e) { err = `invalid JSON: ${e.message}`; }
    }
    $('json-out').textContent = obj ? JSON.stringify(obj, null, 2) : raw;
    $('json-out').dataset.valid = String(!err);
    $('highlight').innerHTML = obj ? highlight(text, obj) : escapeHtml(text);
    const secs = ((performance.now() - t0) / 1000).toFixed(1);
    setStatus(err ? `Done in ${secs}s, but the output is not valid: ${err}` : `Done in ${secs}s: valid JSON.`, 'done');
  } catch (e) {
    setStatus(`Error: ${e.message || e}`, 'error');
  } finally {
    $('extract').disabled = false;
  }
}

$('load').onclick = () => loadModel().catch((e) => { setStatus(`Could not load the model: ${e.message || e}`, 'error'); $('load').disabled = false; });
$('extract').onclick = extract;
window.jobxDemo = { loadModel, extract };
