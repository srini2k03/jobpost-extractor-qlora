// Field-level evidence in the browser: a port of jobx/evidence.py (same statuses and match types).
// Every span is a slice of the posting (start/end offsets into it); a value with no support is "unsupported".
// Salary and experience use number matching here (the Python version uses the full parsers), so the
// browser can report fewer "parsed" matches than the API for the same output.

const BOUND = /[A-Za-z0-9]/;
const FUZZY_MIN = 0.85;
const LEGAL = /\b(pvt|private|ltd|limited|llp|inc|incorporated|corp|corporation|co|company|plc|llc|gmbh)\b\.?/gi;
const SENIORITY_WORDS = /\b(senior|sr|junior|jr|trainee|intern|internship|lead|team lead|mid-level|mid level|mid|entry-level|entry level|managerial|manager level|ii)\b\.?/gi;
const SENIORITY_CUES = {
  intern: /\bintern(ship)?\b/gi,
  junior: /\b(junior|jr\.?|trainee|entry[- ]level|fresher)\b/gi,
  mid: /\b(mid[- ]?(level|senior)?|intermediate|II)\b/gi,
  senior: /\b(senior|sr\.?)\b/gi,
  lead: /\b(lead|principal|staff)\b/gi,
  manager: /\b(manager|managerial|head of)\b/gi,
};
const ALIAS_FIELD = { skills: 'skills', locations: 'locations', work_mode: 'work_mode', employment_type: 'employment_type', education: 'education' };
const WORDS = { zero: 0, one: 1, two: 2, three: 3, four: 4, five: 5, six: 6, seven: 7, eight: 8, nine: 9, ten: 10, twelve: 12, fifteen: 15 };

const esc = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
const span = (text, s, e) => ({ start: s, end: e, text: text.slice(s, e) });

export function tokens(text) {
  const out = [];
  for (const m of text.matchAll(/[A-Za-z0-9+#]+/g)) out.push({ t: m[0].toLowerCase(), s: m.index, e: m.index + m[0].length });
  return out;
}

export function findExact(text, value, caseSensitive = false) {
  const v = String(value).trim();
  if (v.length < 2) return [];
  const re = new RegExp(esc(v), caseSensitive ? 'g' : 'gi');
  const out = [];
  for (const m of text.matchAll(re)) {
    const s = m.index;
    const e = s + m[0].length;
    if (BOUND.test(text[s - 1] || ' ') || BOUND.test(text[e] || ' ')) continue;
    out.push(span(text, s, e));
  }
  return out;
}

export function findTokenSeq(text, value, toks = tokens(text)) {
  const want = tokens(String(value)).map((x) => x.t);
  if (!want.length) return [];
  const out = [];
  for (let i = 0; i + want.length <= toks.length; i += 1) {
    if (want.every((w, j) => toks[i + j].t === w)) out.push(span(text, toks[i].s, toks[i + want.length - 1].e));
  }
  return out;
}

function similarity(a, b) {
  if (a === b) return 1;
  const m = a.length;
  const n = b.length;
  if (!m || !n) return 0;
  let prev = Array.from({ length: n + 1 }, (_, j) => j);
  for (let i = 1; i <= m; i += 1) {
    const cur = [i];
    for (let j = 1; j <= n; j += 1) cur[j] = Math.min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1));
    prev = cur;
  }
  return 1 - prev[n] / Math.max(m, n);
}

export function findFuzzy(text, value, toks = tokens(text)) {
  const target = tokens(String(value)).map((x) => x.t).join(' ');
  if (target.length < 4) return [null, 0];
  const k = target.split(' ').length;
  let best = [0, -1, -1];
  for (const w of new Set([Math.max(1, k - 1), k, k + 1])) {
    for (let i = 0; i + w <= toks.length; i += 1) {
      const cand = toks.slice(i, i + w).map((x) => x.t).join(' ');
      if (Math.abs(cand.length - target.length) > Math.max(3, Math.floor(target.length / 3))) continue;
      const r = similarity(cand, target);
      if (r > best[0]) best = [r, toks[i].s, toks[i + w - 1].e];
    }
  }
  return best[0] >= FUZZY_MIN ? [span(text, best[1], best[2]), best[0]] : [null, best[0]];
}

function aliasSpans(text, field, value, aliases) {
  const list = aliases?.[ALIAS_FIELD[field]]?.[value] || [];
  const cs = new Set(aliases?.case_sensitive?.[ALIAS_FIELD[field]] || []);
  const out = [];
  for (const a of [...list].sort((x, y) => y.length - x.length)) {
    for (const sp of findExact(text, a, cs.has(a))) if (!out.some((o) => sp.start < o.end && o.start < sp.end)) out.push(sp);
  }
  return out;
}

function stringEvidence(text, field, value, toks, aliases) {
  const ev = (status, match, spans, score) => ({ field, value, status, match, spans, score });
  if (ALIAS_FIELD[field]) {
    const hits = aliasSpans(text, field, value, aliases);
    if (hits.length) return ev('supported', hits.some((h) => h.text.toLowerCase() === String(value).toLowerCase()) ? 'exact' : 'alias', hits, 1);
  }
  const ex = findExact(text, value);
  if (ex.length) return ev('supported', 'exact', ex, 1);
  const variants = [value];
  if (field === 'company') variants.push(String(value).replace(LEGAL, ' '));
  if (field === 'role') variants.push(String(value).replace(SENIORITY_WORDS, ' '));
  for (const v of variants) {
    const sp = findTokenSeq(text, v, toks);
    if (sp.length) return ev('supported', 'normalised', sp, 1);
  }
  const [fz, r] = findFuzzy(text, value, toks);
  if (fz) return ev('weak', 'fuzzy', [fz], r);
  return ev('unsupported', 'none', [], r);
}

function numberSpans(text, wanted) {
  const out = [];
  for (const m of text.matchAll(/\d{1,3}(?:,\d{2,3})+(?:\.\d+)?|\d+(?:\.\d+)?/g)) {
    const prev = text[m.index - 1] || ' ';
    if (/[\d.,]/.test(prev)) continue;
    const v = parseFloat(m[0].replace(/,/g, ''));
    if (wanted.some((w) => Math.abs(v - w) < 1e-6)) out.push(span(text, m.index, m.index + m[0].length));
  }
  for (const m of text.matchAll(/\b(zero|one|two|three|four|five|six|seven|eight|nine|ten|twelve|fifteen)\b/gi)) {
    if (wanted.includes(WORDS[m[0].toLowerCase()])) out.push(span(text, m.index, m.index + m[0].length));
  }
  return out;
}

function salaryEvidence(text, sal) {
  const bounds = ['min', 'max'].map((k) => sal?.[k]).filter((v) => typeof v === 'number');
  const found = [];
  let hit = 0;
  for (const b of new Set(bounds)) {
    const cands = [b, b / 1e3, b / 1e5, b / 1e7].map((x) => Math.round(x * 100) / 100);
    const sp = numberSpans(text, cands);
    if (sp.length) { hit += 1; found.push(...sp.slice(0, 2)); }
  }
  const n = new Set(bounds).size;
  const status = n && hit === n ? 'supported' : hit ? 'weak' : 'unsupported';
  return { field: 'salary', value: sal, status, match: status === 'supported' ? 'parsed' : hit ? 'partial' : 'none', spans: found, score: n ? hit / n : 0 };
}

function experienceEvidence(text, field, v) {
  let sp = numberSpans(text, [v]).filter((s) => /^\s*(?:\+|-|–|to)?\s*\d*\s*\+?\s*(?:years?|yrs?|yr|saal)/i.test(text.slice(s.end, s.end + 18))
    || /(?:exp(?:erience)?|min(?:imum)?|at least)[^\n\d]{0,12}$/i.test(text.slice(Math.max(0, s.start - 20), s.start)));
  if (!sp.length && v === 0) sp = [...text.matchAll(/\bfreshers?\b/gi)].map((m) => span(text, m.index, m.index + m[0].length));
  return { field, value: v, status: sp.length ? 'supported' : 'unsupported', match: sp.length ? 'parsed' : 'none', spans: sp.slice(0, 2), score: sp.length ? 1 : 0 };
}

export function findEvidence(text, rec, aliases) {
  const out = [];
  if (!rec || typeof rec !== 'object') return out;
  const toks = tokens(text);
  for (const [field, v] of Object.entries(rec)) {
    if (v == null || v === '' || (Array.isArray(v) && !v.length)) continue;
    if (field === 'skills' || field === 'locations') {
      for (const el of Array.isArray(v) ? v : [v]) if (typeof el === 'string' && el.trim()) out.push(stringEvidence(text, field, el, toks, aliases));
    } else if (field === 'salary' && typeof v === 'object') {
      out.push(salaryEvidence(text, v));
    } else if (field === 'min_experience_years' || field === 'max_experience_years') {
      if (typeof v === 'number') out.push(experienceEvidence(text, field, v));
    } else if (field === 'seniority' && typeof v === 'string') {
      const re = SENIORITY_CUES[v];
      const sp = re ? [...text.matchAll(re)].slice(0, 3).map((m) => span(text, m.index, m.index + m[0].length)) : [];
      out.push({ field, value: v, status: sp.length ? 'supported' : 'unsupported', match: sp.length ? 'alias' : 'none', spans: sp, score: sp.length ? 1 : 0 });
    } else if (typeof v === 'string') {
      out.push(stringEvidence(text, field, v, toks, aliases));
    }
  }
  return out;
}

// For null fields: text that suggests the field was stated (a hint for possible omissions, not proof).
export function missingCues(text, rec, aliases) {
  const cues = {};
  const empty = (k) => rec?.[k] == null || (Array.isArray(rec[k]) && !rec[k].length);
  if (empty('salary')) {
    const m = text.match(/(?:₹|rs\.?|inr|ctc|salary|stipend|package)[^\n]{0,12}?\d[\d,.]*\s*(?:-|to)?\s*[\d,.]*\s*(?:lpa|lakhs?|lacs?|l|k|cr)?\b|\d[\d,.]*\s*(?:-|to)\s*[\d,.]+\s*(?:lpa|lakhs?|lacs?)\b|\d[\d,.]*\s*(?:lpa|lakhs?|lacs?)\b/i);
    if (m) cues.salary = m[0].trim();
  }
  if (empty('min_experience_years') && empty('max_experience_years')) {
    const m = text.match(/\b\d{1,2}\s*(?:-|–|to)?\s*\d{0,2}\s*\+?\s*(?:years?|yrs?)\b|\bfreshers?\b/i);
    if (m) cues.experience = m[0].trim();
  }
  for (const f of ['skills', 'locations', 'work_mode', 'employment_type', 'education']) {
    if (!empty(f)) continue;
    for (const [canon, list] of Object.entries(aliases?.[f] || {})) {
      const hit = list.map((a) => findExact(text, a)).find((x) => x.length);
      if (hit) { cues[f] = `${hit[0].text} → ${canon}`; break; }
    }
  }
  return cues;
}

export function summarise(evs) {
  const by = { supported: 0, weak: 0, unsupported: 0 };
  for (const e of evs) by[e.status] += 1;
  return { n_values: evs.length, ...by };
}
