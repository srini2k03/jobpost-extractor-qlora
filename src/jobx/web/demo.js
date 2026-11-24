// In-browser demo: the fine-tuned Qwen2.5-0.5B LoRA (GGUF Q4_K_M) running locally via wllama (llama.cpp WASM).
// Nothing is sent to a server; the model is downloaded once from this site and cached by the browser.
const WLLAMA = 'https://cdn.jsdelivr.net/npm/@wllama/wllama@__WLLAMA_VERSION__/esm/index.js';
const WASM = { default: 'https://cdn.jsdelivr.net/npm/@wllama/wllama@__WLLAMA_VERSION__/src/wasm/wllama.wasm' };
// Fetch the runtime in the background. The rest of the page (examples, editing) works even if the CDN is
// unreachable; loading the model then reports the error.
const runtime = import(WLLAMA);
runtime.catch(() => {});

const $ = (id) => document.getElementById(id);
const load = async (p) => (await fetch(p)).json();

const [manifest, prompt, schema, aliases, examples] = await Promise.all(
  ['model/manifest.json', 'demo/prompt.json', 'demo/schema.json', 'demo/aliases.json', 'demo/examples.json'].map(load),
);
const KEYS = ['company', 'role', 'seniority', 'skills', 'min_experience_years', 'max_experience_years', 'salary',
  'locations', 'work_mode', 'employment_type', 'education'];
// Highlight label (mark class) -> JSON key, so a field can be traced in both panes.
const LABEL_KEY = { company: 'company', role: 'role', skill: 'skills', location: 'locations', work_mode: 'work_mode',
  employment_type: 'employment_type', education: 'education' };
let wllama = null;

$('size-note').textContent = `${manifest.size_mb} MB, downloaded once and cached by your browser`;

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
}

// ---------- left pane: examples, editing and the highlight view ----------
function showView(view) {
  const marks = view === 'marks';
  $('posting').hidden = marks;
  $('marks-view').hidden = !marks;
  $('view-edit').setAttribute('aria-pressed', String(!marks));
  $('view-marks').setAttribute('aria-pressed', String(marks));
}

function markExample(i) {
  document.querySelectorAll('#examples [data-example]').forEach((b) =>
    b.setAttribute('aria-pressed', String(b.dataset.example === String(i))));
}

examples.forEach((ex, i) => {
  const b = document.createElement('button');
  b.type = 'button';
  b.textContent = `Example ${i + 1}`;
  b.className = 'jx-chip';
  b.title = ex.text.split('\n')[0];
  b.dataset.example = String(i);
  b.onclick = () => { $('posting').value = ex.text; markExample(i); showView('edit'); };
  $('examples').appendChild(b);
});
$('posting').value = examples[0]?.text || '';
markExample(0);
$('posting').addEventListener('input', () => markExample(-1));
$('view-edit').onclick = () => { showView('edit'); $('posting').focus(); };
$('view-marks').onclick = () => showView('marks');

// ---------- status, steps and progress ----------
function setStatus(text, state) {
  $('status').textContent = text;
  $('status').dataset.state = state || '';
}

function setProgress(pct) {
  $('progress-bar').style.width = `${pct}%`;
  $('progress-pct').textContent = `${pct}%`;
}

function setStep(step, cls) {
  const el = $(step);
  el.classList.remove('is-active', 'is-done');
  if (cls) el.classList.add(cls);
}

// ---------- right pane: the JSON ----------
function skeleton(busy) {
  const widths = [9, 11, 6, 14, 2, 2, 12, 10, 6, 8, 6];
  $('json-out').innerHTML = '<span class="sr-only">No output yet.</span><span aria-hidden="true">{\n' +
    KEYS.map((k, i) => `  <span class="jx-key-faint">"${k}"</span>: <span class="jx-skel" style="--sw:${widths[i]}ch"></span>`)
      .join(',\n') + '\n}</span>';
  $('json-out').classList.toggle('is-busy', !!busy);
}

function colour(json) {
  return escapeHtml(json).replace(
    /(&quot;(?:[^&\\]|\\.|&(?!quot;))*&quot;)(\s*:)?|\b(true|false|null)\b|-?\b\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\b/g,
    (m, str, colon, lit) => {
      if (str) return colon ? `<span class="k">${str}</span>${colon}` : `<span class="s">${str}</span>`;
      if (lit) return `<span class="${lit === 'null' ? 'null' : 'n'}">${lit}</span>`;
      return `<span class="n">${m}</span>`;
    },
  );
}

// Same text as JSON.stringify(obj, null, 2); each top-level field is wrapped so it can be traced in the posting.
function renderJson(obj) {
  const keys = Object.keys(obj);
  if (!keys.length) return '{}';
  const lines = keys.map((k) => {
    const v = JSON.stringify(obj[k], null, 2).replace(/\n/g, '\n  ');
    return `<span class="jl" data-f="${escapeHtml(k)}">  ${colour(`${JSON.stringify(k)}:`)} ${colour(v)}</span>`;
  });
  return `{\n${lines.join(',\n')}\n}`;
}

function validate(obj) {
  if (typeof obj !== 'object' || obj === null || Array.isArray(obj)) return 'not a JSON object';
  const missing = KEYS.filter((k) => !(k in obj));
  return missing.length ? `missing keys: ${missing.join(', ')}` : null;
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
      `<mark class="f-${s.label}" data-f="${LABEL_KEY[s.label]}" title="${s.label}">${escapeHtml(text.slice(s.start, s.end))}</mark>`;
    pos = s.end;
  }
  return html + escapeHtml(text.slice(pos));
}

// Hovering a JSON field, a highlighted value or a legend entry lights up the same field in both panes.
function setHot(key) {
  const hl = $('highlight');
  hl.classList.toggle('has-hot', !!key && !!hl.querySelector(`mark[data-f="${key}"]`));
  hl.querySelectorAll('mark').forEach((m) => m.classList.toggle('is-hot', m.dataset.f === key));
  $('json-out').querySelectorAll('.jl').forEach((l) => l.classList.toggle('is-hot', l.dataset.f === key));
}
$('json-out').addEventListener('pointerover', (e) => setHot(e.target.closest('.jl')?.dataset.f || null));
$('json-out').addEventListener('pointerleave', () => setHot(null));
$('highlight').addEventListener('pointerover', (e) => setHot(e.target.closest('mark')?.dataset.f || null));
$('highlight').addEventListener('pointerleave', () => setHot(null));
document.querySelectorAll('.jx-legend__item').forEach((b) => {
  const key = LABEL_KEY[b.dataset.f];
  b.addEventListener('pointerenter', () => setHot(key));
  b.addEventListener('pointerleave', () => setHot(null));
  b.addEventListener('focus', () => setHot(key));
  b.addEventListener('blur', () => setHot(null));
});

// ---------- model ----------
async function loadModel() {
  if (wllama) return;
  $('load').disabled = true;
  setStep('step-load', 'is-active');
  setStatus('Loading the model…', 'loading');
  const { Wllama } = await runtime;
  const w = new Wllama(WASM);
  // The GGUF is served as byte chunks (GitHub Pages file-size limit); download and join them.
  const total = manifest.size_mb * 1e6;
  let loaded = 0;
  const chunks = [];
  for (const part of manifest.parts) {
    const res = await fetch(new URL(part, location.href).href);
    if (!res.ok) throw new Error(`${part}: HTTP ${res.status}`);
    const reader = res.body.getReader();
    const bufs = [];
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      bufs.push(value);
      loaded += value.length;
      const pct = Math.min(100, Math.round((100 * loaded) / total));
      setProgress(pct);
      setStatus(`Downloading ${pct}% (${Math.round(loaded / 1e6)} of ${manifest.size_mb} MB)`, 'loading');
    }
    chunks.push(new Blob(bufs));
  }
  setProgress(100);
  setStatus('Starting the model…', 'loading');
  const gguf = new File(chunks, manifest.file || 'model.gguf');
  await w.loadModel([gguf], { n_ctx: 2048 });
  wllama = w;
  $('load').textContent = 'Model loaded';
  $('load').classList.remove('sb-btn--primary');
  $('extract').disabled = false;
  $('extract').classList.add('sb-btn--primary');
  setStep('step-load', 'is-done');
  setStep('step-extract', 'is-active');
  setStatus('Model ready. Paste a posting or pick an example, then Extract.', 'ready');
}

async function extract() {
  const text = $('posting').value.trim();
  if (!text) return;
  $('extract').disabled = true;
  $('valid-badge').hidden = true;
  setHot(null);
  skeleton(true);
  setStatus('Extracting…', 'busy');
  const t0 = performance.now();
  const tick = setInterval(() => {
    setStatus(`Extracting… ${Math.round((performance.now() - t0) / 1000)} s`, 'busy');
  }, 1000);
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
    clearInterval(tick);
    const out = $('json-out');
    out.classList.remove('is-busy', 'is-fresh');
    if (obj && typeof obj === 'object' && !Array.isArray(obj)) out.innerHTML = renderJson(obj);
    else out.textContent = obj ? JSON.stringify(obj, null, 2) : raw;
    void out.offsetWidth;
    out.classList.add('is-fresh');
    out.dataset.valid = String(!err);
    $('highlight').innerHTML = obj ? highlight(text, obj) : escapeHtml(text);
    $('view-marks').disabled = false;
    showView('marks');
    const badge = $('valid-badge');
    badge.hidden = false;
    badge.className = err ? 'sb-pill sb-pill--pending' : 'sb-pill sb-pill--live';
    badge.textContent = err ? 'not valid' : `valid · ${KEYS.length} fields`;
    const secs = ((performance.now() - t0) / 1000).toFixed(1);
    setStep('step-extract', 'is-done');
    setStatus(err ? `Done in ${secs}s, but the output is not valid: ${err}` : `Done in ${secs}s: valid JSON.`, 'done');
  } catch (e) {
    clearInterval(tick);
    skeleton(false);
    setStatus(`Error: ${e.message || e}`, 'error');
  } finally {
    $('extract').disabled = false;
  }
}

skeleton(false);
setStep('step-load', 'is-active');
$('load').onclick = () => loadModel().catch((e) => {
  setStatus(`Could not load the model: ${e.message || e}`, 'error');
  setStep('step-load', 'is-active');
  $('load').disabled = false;
});
$('extract').onclick = extract;
window.jobxDemo = { loadModel, extract };
