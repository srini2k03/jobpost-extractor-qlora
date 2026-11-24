// In-browser demo: the fine-tuned Qwen2.5-0.5B LoRA (GGUF Q4_K_M) running locally via wllama (llama.cpp WASM).
// Nothing is sent to a server; the model is downloaded once from this site and cached by the browser.
import { findEvidence, missingCues, summarise } from './evidence.js';

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
const ENUMS = {
  seniority: ['intern', 'junior', 'mid', 'senior', 'lead', 'manager'],
  work_mode: ['onsite', 'hybrid', 'remote'],
  employment_type: ['full_time', 'part_time', 'contract', 'internship', 'freelance'],
  education: ['10th', '12th', 'diploma', 'graduate', 'b.tech', 'bca', 'b.sc', 'b.com', 'bba', 'mca', 'm.tech', 'm.sc',
    'mba', 'ca', 'phd'],
};
// Highlight label (mark class) -> JSON key, so a field can be traced in both panes.
const LABEL_KEY = { company: 'company', role: 'role', skill: 'skills', location: 'locations', work_mode: 'work_mode',
  employment_type: 'employment_type', education: 'education', salary: 'salary', experience: 'min_experience_years',
  seniority: 'seniority' };
const KEY_LABEL = { skills: 'skill', locations: 'location', min_experience_years: 'experience',
  max_experience_years: 'experience' };
// The model was loaded with a 2,048-token context; the prompt and the JSON answer take ~500 of them.
const N_CTX = 2048;
const SOFT_LIMIT = 5000; // characters: above this a posting may not fit the context
const HARD_LIMIT = 12000; // characters: refuse (keeps the tab responsive)
let wllama = null;
let controller = null;
let lastResult = null;

$('size-note').textContent = `${manifest.size_mb} MB, downloaded once and cached by your browser`;

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
}

// ---------- device check ----------
(function memoryCheck() {
  const el = $('mem-note');
  if (!el) return;
  const gb = navigator.deviceMemory; // Chromium only, rounded, capped at 8
  const need = Math.max(0.6, (manifest.size_mb / 1000) * 1.4);
  if (typeof gb === 'number') {
    el.textContent = gb < 2 ? `Your device reports ~${gb} GB of RAM; the model needs ~${need.toFixed(1)} GB. Loading may fail.`
      : `~${gb} GB RAM reported · needs ~${need.toFixed(1)} GB`;
    el.dataset.level = gb < 2 ? 'warn' : 'ok';
  } else {
    el.textContent = `The model needs ~${need.toFixed(1)} GB of free memory (your browser does not report device memory).`;
  }
  if (navigator.storage?.estimate) {
    navigator.storage.estimate().then(({ quota, usage }) => {
      if (quota && quota - (usage || 0) < manifest.size_mb * 1e6 * 1.2) {
        el.textContent += ' Browser storage is low, so the model may not be cached.';
        el.dataset.level = 'warn';
      }
    }).catch(() => {});
  }
})();

// ---------- left pane: examples, file input, editing and the highlight view ----------
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

function updateCount() {
  const n = $('posting').value.length;
  const el = $('char-count');
  if (!el) return;
  el.textContent = `${n.toLocaleString('en-IN')} chars`;
  el.dataset.level = n > HARD_LIMIT ? 'error' : n > SOFT_LIMIT ? 'warn' : '';
  el.title = n > SOFT_LIMIT ? `Long posting: the model reads ~${N_CTX} tokens in total, so the end may be cut off.` : '';
}

function setPosting(text) {
  $('posting').value = text;
  updateCount();
  showView('edit');
}

examples.forEach((ex, i) => {
  const b = document.createElement('button');
  b.type = 'button';
  b.textContent = ex.label || `Example ${i + 1}`;
  b.className = 'jx-chip';
  b.title = ex.text.split('\n')[0];
  b.dataset.example = String(i);
  b.onclick = () => { setPosting(ex.text); markExample(i); };
  $('examples').appendChild(b);
});
setPosting(examples[0]?.text || '');
markExample(0);
$('posting').addEventListener('input', () => { markExample(-1); updateCount(); });
$('view-edit').onclick = () => { showView('edit'); $('posting').focus(); };
$('view-marks').onclick = () => showView('marks');

function htmlToText(html) {
  const doc = new DOMParser().parseFromString(html, 'text/html');
  doc.querySelectorAll('script, style, noscript, template').forEach((n) => n.remove());
  doc.querySelectorAll('br, p, div, li, h1, h2, h3, h4, h5, h6, tr').forEach((n) => n.append('\n'));
  return (doc.body?.textContent || '').replace(/[ \t]+\n/g, '\n').replace(/\n{3,}/g, '\n\n').trim();
}

$('file-in')?.addEventListener('change', async (e) => {
  const f = e.target.files?.[0];
  e.target.value = '';
  if (!f) return;
  if (f.size > 1_000_000) { setStatus(`${f.name} is larger than 1 MB; paste the posting text instead.`, 'error'); return; }
  const ext = (f.name.split('.').pop() || '').toLowerCase();
  if (!['txt', 'md', 'markdown', 'html', 'htm'].includes(ext)) {
    setStatus(`Unsupported file type .${ext}: use .txt, .md or .html.`, 'error');
    return;
  }
  const raw = await f.text();
  const text = ext === 'html' || ext === 'htm' ? htmlToText(raw) : raw.trim();
  if (!text) { setStatus(`${f.name} has no text.`, 'error'); return; }
  setPosting(text);
  markExample(-1);
  setStatus(`Loaded ${f.name} (${text.length.toLocaleString('en-IN')} characters).`, wllama ? 'ready' : '');
});

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

// Schema check mirroring the Pydantic model (keys, types, enums, ranges).
function validate(obj) {
  if (typeof obj !== 'object' || obj === null || Array.isArray(obj)) return ['not a JSON object'];
  const errs = [];
  const missing = KEYS.filter((k) => !(k in obj));
  if (missing.length) errs.push(`missing keys: ${missing.join(', ')}`);
  const extra = Object.keys(obj).filter((k) => !KEYS.includes(k));
  if (extra.length) errs.push(`unexpected keys: ${extra.join(', ')}`);
  for (const k of ['company', 'role']) if (obj[k] != null && typeof obj[k] !== 'string') errs.push(`${k} must be a string or null`);
  for (const k of ['skills', 'locations']) {
    if (!Array.isArray(obj[k]) || obj[k].some((x) => typeof x !== 'string')) errs.push(`${k} must be a list of strings`);
  }
  for (const [k, vals] of Object.entries(ENUMS)) if (obj[k] != null && !vals.includes(obj[k])) errs.push(`${k} "${obj[k]}" is not allowed`);
  const lo = obj.min_experience_years;
  const hi = obj.max_experience_years;
  for (const [k, v] of [['min_experience_years', lo], ['max_experience_years', hi]]) {
    if (v != null && (typeof v !== 'number' || v < 0 || v > 50)) errs.push(`${k} must be a number from 0 to 50 or null`);
  }
  if (typeof lo === 'number' && typeof hi === 'number' && lo > hi) errs.push('min_experience_years > max_experience_years');
  const s = obj.salary;
  if (s != null) {
    if (typeof s !== 'object' || Array.isArray(s)) errs.push('salary must be an object or null');
    else {
      if (s.min == null && s.max == null) errs.push('salary needs min or max');
      if (typeof s.min === 'number' && typeof s.max === 'number' && s.min > s.max) errs.push('salary.min > salary.max');
      if (!['INR', 'USD'].includes(s.currency)) errs.push('salary.currency must be INR or USD');
      if (!['year', 'month', 'hour'].includes(s.period)) errs.push('salary.period must be year, month or hour');
    }
  }
  return errs;
}

function renderMarks(text, evidence) {
  const spans = [];
  for (const e of evidence) {
    const label = KEY_LABEL[e.field] || e.field;
    for (const sp of e.spans) spans.push({ ...sp, label, key: e.field.endsWith('_experience_years') ? 'min_experience_years' : e.field, weak: e.status === 'weak' });
  }
  spans.sort((a, b) => a.start - b.start || b.end - a.end);
  let html = '';
  let pos = 0;
  for (const s of spans) {
    if (s.start < pos) continue;
    const title = `${s.label}${s.weak ? ' (weak match)' : ''}`;
    html += escapeHtml(text.slice(pos, s.start)) +
      `<mark class="f-${s.label}${s.weak ? ' is-weak' : ''}" data-f="${s.key}" title="${title}">${escapeHtml(text.slice(s.start, s.end))}</mark>`;
    pos = s.end;
  }
  return html + escapeHtml(text.slice(pos));
}

const STATUS_TEXT = { supported: 'found in text', weak: 'weak match', unsupported: 'not found in text', null: 'not stated' };

function inr(n) {
  return Number(n).toLocaleString('en-IN', { maximumFractionDigits: 2 });
}

function fmtValue(v) {
  if (v == null) return 'null';
  if (Array.isArray(v)) return v.join(', ');
  if (typeof v === 'object' && ('min' in v || 'max' in v)) {
    const cur = v.currency === 'USD' ? '$' : '₹';
    const range = v.min != null && v.max != null && v.min !== v.max ? `${cur}${inr(v.min)} – ${cur}${inr(v.max)}`
      : v.min == null ? `up to ${cur}${inr(v.max)}` : v.max == null ? `from ${cur}${inr(v.min)}` : `${cur}${inr(v.min)}`;
    return `${range} per ${v.period || 'year'}`;
  }
  if (typeof v === 'object') return JSON.stringify(v);
  return String(v);
}

function renderFieldCheck(text, obj, evidence) {
  const list = $('field-check');
  if (!list) return;
  const cues = missingCues(text, obj, aliases);
  const rows = KEYS.map((k) => {
    const v = obj[k];
    const empty = v == null || (Array.isArray(v) && !v.length);
    const evs = evidence.filter((e) => e.field === k);
    let status = 'null';
    if (!empty) {
      status = evs.some((e) => e.status === 'unsupported') ? 'unsupported' : evs.some((e) => e.status === 'weak') ? 'weak' : 'supported';
    }
    const snippet = evs.flatMap((e) => e.spans.map((s) => s.text)).filter(Boolean);
    const unsup = evs.filter((e) => e.status === 'unsupported').map((e) => fmtValue(e.value));
    const cueKey = k.endsWith('_experience_years') ? 'experience' : k;
    let detail = snippet.length ? `“${escapeHtml([...new Set(snippet)].slice(0, 3).join('”, “'))}”` : '';
    if (unsup.length && Array.isArray(v)) detail += `${detail ? ' · ' : ''}not in text: ${escapeHtml(unsup.join(', '))}`;
    if (status === 'null' && cues[cueKey]) detail = `possibly stated: “${escapeHtml(cues[cueKey])}”`;
    return `<li class="jx-fc" data-f="${k}" data-status="${status}">
      <span class="jx-fc__k">${k}</span>
      <span class="jx-fc__v" title="${escapeHtml(fmtValue(v))}">${escapeHtml(empty ? '—' : fmtValue(v))}</span>
      <span class="jx-fc__s"><span class="jx-ev jx-ev--${status}">${STATUS_TEXT[status]}</span></span>
      <span class="jx-fc__d">${detail}</span></li>`;
  });
  list.innerHTML = rows.join('');
  const s = summarise(evidence);
  $('fc-summary').textContent = s.n_values
    ? `${s.supported} of ${s.n_values} extracted values found in the posting` +
      (s.weak ? `, ${s.weak} weak` : '') + (s.unsupported ? `, ${s.unsupported} not found (check them)` : '') + '.'
    : 'No values extracted.';
  $('field-check-wrap').hidden = false;
}

// Hovering a JSON field, a highlighted value, a field-check row or a legend entry lights up the field everywhere.
function setHot(key) {
  const hl = $('highlight');
  hl.classList.toggle('has-hot', !!key && !!hl.querySelector(`mark[data-f="${key}"]`));
  hl.querySelectorAll('mark').forEach((m) => m.classList.toggle('is-hot', m.dataset.f === key));
  $('json-out').querySelectorAll('.jl').forEach((l) => l.classList.toggle('is-hot', l.dataset.f === key));
  document.querySelectorAll('#field-check .jx-fc').forEach((r) => r.classList.toggle('is-hot', r.dataset.f === key));
}
$('json-out').addEventListener('pointerover', (e) => setHot(e.target.closest('.jl')?.dataset.f || null));
$('json-out').addEventListener('pointerleave', () => setHot(null));
$('highlight').addEventListener('pointerover', (e) => setHot(e.target.closest('mark')?.dataset.f || null));
$('highlight').addEventListener('pointerleave', () => setHot(null));
$('field-check')?.addEventListener('pointerover', (e) => setHot(e.target.closest('.jx-fc')?.dataset.f || null));
$('field-check')?.addEventListener('pointerleave', () => setHot(null));
document.querySelectorAll('.jx-legend__item').forEach((b) => {
  const key = LABEL_KEY[b.dataset.f];
  b.addEventListener('pointerenter', () => setHot(key));
  b.addEventListener('pointerleave', () => setHot(null));
  b.addEventListener('focus', () => setHot(key));
  b.addEventListener('blur', () => setHot(null));
});

// ---------- copy / download ----------
function setTools(enabled) {
  for (const id of ['copy-json', 'download-json']) if ($(id)) $(id).disabled = !enabled;
}
$('copy-json')?.addEventListener('click', async () => {
  if (!lastResult) return;
  const txt = JSON.stringify(lastResult, null, 2);
  try {
    await navigator.clipboard.writeText(txt);
    $('copy-json').textContent = 'Copied';
  } catch {
    const ta = document.createElement('textarea');
    ta.value = txt;
    document.body.appendChild(ta);
    ta.select();
    const ok = document.execCommand('copy');
    ta.remove();
    $('copy-json').textContent = ok ? 'Copied' : 'Copy failed';
  }
  setTimeout(() => { $('copy-json').textContent = 'Copy JSON'; }, 1600);
});
$('download-json')?.addEventListener('click', () => {
  if (!lastResult) return;
  const blob = new Blob([`${JSON.stringify(lastResult, null, 2)}\n`], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'job-posting.json';
  document.body.appendChild(a);
  a.click();
  setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 0);
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
  const t0 = performance.now();
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
  const dl = (performance.now() - t0) / 1000;
  setStatus('Starting the model…', 'loading');
  const t1 = performance.now();
  const gguf = new File(chunks, manifest.file || 'model.gguf');
  await w.loadModel([gguf], { n_ctx: N_CTX });
  wllama = w;
  const init = (performance.now() - t1) / 1000;
  $('load').textContent = 'Model loaded';
  $('load').classList.remove('sb-btn--primary');
  $('extract').disabled = false;
  $('extract').classList.add('sb-btn--primary');
  setStep('step-load', 'is-done');
  setStep('step-extract', 'is-active');
  setStatus(`Model ready (download ${dl.toFixed(1)} s, start ${init.toFixed(1)} s). Paste a posting or pick an example, then Extract.`, 'ready');
}

function showMetrics(m) {
  const el = $('run-metrics');
  if (!el) return;
  const rows = [
    ['Latency', `${m.latency.toFixed(1)} s`],
    ['Prompt tokens', m.promptTokens ?? '—'],
    ['Output tokens', m.outTokens ?? '—'],
    ['Generation', m.genTps ? `${m.genTps.toFixed(1)} tok/s` : '—'],
    ['Prompt reading', m.ppTps ? `${Math.round(m.ppTps)} tok/s` : '—'],
  ];
  el.innerHTML = rows.map(([k, v]) => `<div><dt>${k}</dt><dd>${v}</dd></div>`).join('');
  el.hidden = false;
}

async function generate(req, onProgress) {
  // Streaming gives live progress; if this wllama build rejects it, fall back to one blocking call.
  let text = '';
  let usage = null;
  let timings = null;
  let n = 0;
  try {
    await wllama.createChatCompletion({
      ...req,
      stream: true,
      onData: (chunk) => {
        const d = chunk?.choices?.[0]?.delta?.content;
        if (d) { text += d; n += 1; onProgress(n, text); }
        if (chunk?.usage) usage = chunk.usage;
        if (chunk?.timings) timings = chunk.timings;
      },
    });
    if (text || n) return { text, usage, timings, n };
  } catch (e) {
    if (e?.name === 'AbortError' || req.abortSignal?.aborted) throw e;
    if (text) throw e;
  }
  const res = await wllama.createChatCompletion(req);
  return { text: res.choices[0].message.content || '', usage: res.usage || null, timings: res.timings || null, n: res.usage?.completion_tokens || 0 };
}

async function extract() {
  const text = $('posting').value.trim();
  if (!text) { setStatus('Paste a job posting or pick an example first.', 'error'); return; }
  if (text.length > HARD_LIMIT) {
    setStatus(`This posting has ${text.length.toLocaleString('en-IN')} characters; the in-browser model reads at most ~${N_CTX} tokens. Trim it to the job description.`, 'error');
    return;
  }
  $('extract').disabled = true;
  $('valid-badge').hidden = true;
  setTools(false);
  setHot(null);
  skeleton(true);
  controller = new AbortController();
  if ($('cancel')) $('cancel').hidden = false;
  setStatus('Extracting… reading the posting', 'busy');
  const t0 = performance.now();
  let tokens = 0;
  const tick = setInterval(() => {
    const s = (performance.now() - t0) / 1000;
    setStatus(tokens ? `Extracting… ${tokens} tokens in ${Math.round(s)} s` : `Extracting… reading the posting (${Math.round(s)} s)`, 'busy');
  }, 500);
  const req = {
    messages: [
      { role: 'system', content: prompt.system },
      { role: 'user', content: prompt.user_template.replace('{text}', text) },
    ],
    max_tokens: 400,
    temperature: 0,
    abortSignal: controller.signal,
  };
  if ($('guided').checked) req.response_format = { type: 'json_schema', json_schema: { name: 'JobPosting', schema } };
  try {
    const out = await generate(req, (n) => { tokens = n; });
    const raw = out.text;
    const m = raw.match(/\{[\s\S]*\}/);
    let obj = null;
    let errs = ['no JSON object in the output'];
    if (m) {
      try { obj = JSON.parse(m[0]); errs = validate(obj); } catch (e) { errs = [`invalid JSON: ${e.message}`]; }
    }
    clearInterval(tick);
    const secs = (performance.now() - t0) / 1000;
    const jsonOut = $('json-out');
    jsonOut.classList.remove('is-busy', 'is-fresh');
    if (obj && typeof obj === 'object' && !Array.isArray(obj)) jsonOut.innerHTML = renderJson(obj);
    else jsonOut.textContent = obj ? JSON.stringify(obj, null, 2) : raw;
    void jsonOut.offsetWidth;
    jsonOut.classList.add('is-fresh');
    jsonOut.dataset.valid = String(!errs.length);
    const isObj = obj && typeof obj === 'object' && !Array.isArray(obj);
    const evidence = isObj ? findEvidence(text, obj, aliases) : [];
    $('highlight').innerHTML = isObj ? renderMarks(text, evidence) : escapeHtml(text);
    if (isObj) renderFieldCheck(text, obj, evidence);
    lastResult = isObj ? obj : null;
    setTools(!!lastResult);
    $('view-marks').disabled = false;
    showView('marks');
    const badge = $('valid-badge');
    badge.hidden = false;
    badge.className = errs.length ? 'sb-pill sb-pill--pending' : 'sb-pill sb-pill--live';
    badge.textContent = errs.length ? 'not valid' : `valid · ${KEYS.length} fields`;
    badge.title = errs.join('; ');
    const tm = out.timings || {};
    showMetrics({
      latency: secs,
      promptTokens: out.usage?.prompt_tokens ?? tm.prompt_n,
      outTokens: out.usage?.completion_tokens ?? (tm.predicted_n || out.n || null),
      genTps: tm.predicted_per_second || null,
      ppTps: tm.prompt_per_second || null,
    });
    setStep('step-extract', 'is-done');
    setStatus(errs.length ? `Done in ${secs.toFixed(1)}s, but the output is not valid: ${errs.join('; ')}` : `Done in ${secs.toFixed(1)}s: valid JSON.`, 'done');
  } catch (e) {
    clearInterval(tick);
    skeleton(false);
    const cancelled = e?.name === 'AbortError' || controller?.signal.aborted;
    setStatus(cancelled ? 'Cancelled.' : `Error: ${e.message || e}`, cancelled ? 'ready' : 'error');
  } finally {
    $('extract').disabled = false;
    if ($('cancel')) $('cancel').hidden = true;
    controller = null;
  }
}

// The CI check (scripts/check_demo.py) runs this page with the real model in headless Chromium before every
// publish and saves what it measured; show it so visitors know what to expect.
fetch('browser_check.json').then((r) => (r.ok ? r.json() : null)).then((c) => {
  if (!c || !$('ci-note')) return;
  const m = c.metrics || {};
  $('ci-note').textContent = `Last CI run (${c.date}, ${c.browser}): model ready after ${c.load_s} s, extraction ${c.extract_s} s` +
    `${m.Generation ? `, ${m.Generation}` : ''}${m['Output tokens'] ? ` for ${m['Output tokens']} output tokens` : ''}, ${c.valid ? 'valid JSON' : 'invalid output'}.`;
  $('ci-note').hidden = false;
}).catch(() => {});

skeleton(false);
setTools(false);
setStep('step-load', 'is-active');
$('load').onclick = () => loadModel().catch((e) => {
  setStatus(`Could not load the model: ${e.message || e}`, 'error');
  setStep('step-load', 'is-active');
  $('load').disabled = false;
});
$('extract').onclick = extract;
if ($('cancel')) $('cancel').onclick = () => controller?.abort();
window.jobxDemo = { loadModel, extract, validate };
