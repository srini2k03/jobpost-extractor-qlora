// Base vs fine-tuned: precomputed outputs from committed evaluation runs (llama.cpp in GitHub Actions).
// Nothing here runs a model; "Try it live" copies the posting into the in-browser demo above.
const $ = (id) => document.getElementById(id);
const data = await (await fetch('demo/compare.json')).json();
const KEYS = ['company', 'role', 'seniority', 'skills', 'min_experience_years', 'max_experience_years', 'salary',
  'locations', 'work_mode', 'employment_type', 'education'];

function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
}

function fmt(v) {
  if (v == null || (Array.isArray(v) && !v.length)) return '<span class="jx-null">null</span>';
  if (Array.isArray(v)) return esc(v.join(', '));
  if (typeof v === 'object') {
    const cur = v.currency === 'USD' ? '$' : '₹';
    const n = (x) => Number(x).toLocaleString('en-IN');
    const r = v.min != null && v.max != null && v.min !== v.max ? `${cur}${n(v.min)}–${n(v.max)}`
      : v.min == null ? `≤ ${cur}${n(v.max)}` : v.max == null ? `≥ ${cur}${n(v.min)}` : `${cur}${n(v.min)}`;
    return esc(`${r} / ${v.period}`);
  }
  return esc(v);
}

function cell(side, k) {
  if (!side.json_ok) return '<td class="jx-cmp__c is-bad"><span class="sr-only">wrong: </span>no valid JSON</td>';
  const ok = side.correct[k];
  return `<td class="jx-cmp__c ${ok ? 'is-ok' : 'is-bad'}"><span class="jx-cmp__mark" aria-hidden="true">${ok ? '✓' : '✗'}</span>` +
    `<span class="sr-only">${ok ? 'correct' : 'wrong'}: </span>${fmt(side.data[k])}</td>`;
}

function show(i) {
  const ex = data.examples[i];
  document.querySelectorAll('#cmp-picks [data-i]').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.i === String(i))));
  $('cmp-text').textContent = ex.text;
  $('cmp-src').textContent = `${ex.id} · ${ex.source || ''}${ex.licence ? ` (${ex.licence})` : ''}`;
  const nb = KEYS.filter((k) => ex.base.json_ok && ex.base.correct[k]).length;
  const no = KEYS.filter((k) => ex.ours.json_ok && ex.ours.correct[k]).length;
  $('cmp-score').innerHTML = `Fields matching the label: base <strong>${nb}/11</strong> &middot; fine-tuned <strong>${no}/11</strong>`;
  $('cmp-body').innerHTML = KEYS.map((k) => `<tr><th scope="row">${k}</th><td class="jx-cmp__g">${fmt(ex.gold[k])}</td>${cell(ex.base, k)}${cell(ex.ours, k)}</tr>`).join('');
}

data.examples.forEach((ex, i) => {
  const b = document.createElement('button');
  b.type = 'button';
  b.className = 'jx-chip';
  b.dataset.i = String(i);
  b.textContent = `Posting ${i + 1}`;
  b.title = ex.text.split('\n')[0];
  b.onclick = () => show(i);
  $('cmp-picks').appendChild(b);
});
$('cmp-try').onclick = () => {
  const i = Number(document.querySelector('#cmp-picks [aria-pressed="true"]')?.dataset.i || 0);
  const ta = $('posting');
  if (!ta) return;
  ta.value = data.examples[i].text;
  ta.dispatchEvent(new Event('input'));
  $('view-edit')?.click();
  document.getElementById('demo')?.scrollIntoView({ behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth' });
};
if (data.examples.length) show(0);
