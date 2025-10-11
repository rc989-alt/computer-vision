#!/usr/bin/env node
/**
 * Compare two inspirations files (e.g., baseline vs YOLO-grounded) and emit a Markdown report.
 * Usage:
 *   node run_dual_ablation_report.mjs \
 *     --base api/image_service/data/inspirations_from_precompute.json \
 *     --variant api/image_service/data/inspirations_from_yolo.json \
 *     --out api/image_service/data/reports/ablation_yolo_vs_base.md \
 *     [--top-k 1]
 */
import fs from 'node:fs';
import fsp from 'node:fs/promises';
import path from 'node:path';

function arg(name, defVal = null) {
  const i = process.argv.indexOf(name);
  if (i === -1) return defVal;
  return process.argv[i + 1] ?? true;
}

const BASE = arg('--base', 'api/image_service/data/inspirations_from_precompute.json');
const VARIANT = arg('--variant', 'api/image_service/data/inspirations_from_yolo.json');
const OUT = arg('--out', `api/image_service/data/reports/ablation_${new Date().toISOString().slice(0,10)}_compare.md`);
const TOPK = parseInt(arg('--top-k', '1'), 10) || 1;

function readJson(p){ return JSON.parse(fs.readFileSync(p, 'utf-8')); }

function classifyIntent(garnish){
  const s = String(garnish || '').toLowerCase();
  return {
    isFlower: /(blossom|flower|petal|sakura)/.test(s) || /cherry\s+blossom/.test(s),
    isCherryFruit: /\bcherry\b/.test(s) && !(/(blossom|flower|petal|sakura)/.test(s)),
    isZest: /(zest|twist|peel|strip)/.test(s),
    isChunk: /(wedge|slice|wheel|segment)/.test(s),
    fruit: (/(lime|orange|lemon|grapefruit|cherry)/.exec(s) || [null,null])[1]
  };
}

function textSignals(txt){
  const t = String(txt || '').toLowerCase();
  return {
    blossom: /(blossom|flower|petal|sakura)/.test(t) || /cherry\s+blossom/.test(t),
    cherry: /\bmaraschino\b|cocktail\s+cherry|\bcherry\b/.test(t),
    zest: /(zest|twist|peel|strip)/.test(t),
    chunk: /(wedge|slice|wheel|segment)/.test(t),
    lime: /\blime\b/.test(t),
    orange: /\borange\b/.test(t),
    lemon: /\blemon\b/.test(t),
    grapefruit: /\bgrapefruit\b/.test(t)
  };
}

function topKCompliance(insp, k){
  const intent = classifyIntent(insp?.cocktail?.visual?.garnish);
  const cands = (insp?.candidates || []).slice(0, k);
  if (!cands.length) return { compliant: false, conflicts: 0 };
  let okAny = false, conflicts = 0;
  for (const c of cands) {
    const txt = (c.alt_description || '') + ' ' + (c.description || '');
    const s = textSignals(txt);
    let ok = 0, bad = 0;
    if (intent.isFlower) { if (s.blossom) ok++; if (s.cherry && !s.blossom) bad++; }
    if (intent.isCherryFruit) { if (s.cherry) ok++; if (s.blossom) bad++; }
    if (intent.isZest) { if (s.zest) ok++; if (s.chunk) bad++; }
    if (intent.isChunk) { if (s.chunk) ok++; if (s.zest) bad++; }
    if (intent.fruit) { if (s[intent.fruit]) ok++; }
    okAny = okAny || (ok > 0);
    if (bad > 0) conflicts += 1;
  }
  return { compliant: okAny, conflicts };
}

function aggregate(data, k){
  let total=0, compliant=0, conflicts=0;
  for(const rec of data){
    for(const insp of (rec.inspirations||[])){
      const r = topKCompliance(insp, k);
      total += 1; compliant += r.compliant ? 1 : 0; conflicts += r.conflicts;
    }
  }
  return { total, compliant, rate: total? compliant/total : 0, conflicts };
}

async function main(){
  const base = readJson(BASE);
  const varr = readJson(VARIANT);
  const a = aggregate(base, TOPK);
  const b = aggregate(varr, TOPK);
  const lines = [];
  lines.push(`# Dual Ablation Report`);
  lines.push('');
  lines.push(`Top-K: ${TOPK}`);
  lines.push('');
  lines.push('| Variant | Compliance@K | Conflicts | Notes |');
  lines.push('|---|---:|---:|---|');
  lines.push(`| Baseline (retrieval-only) | ${(a.rate*100).toFixed(1)}% | ${a.conflicts} | inspirations_from_precompute.json |`);
  lines.push(`| + YOLO grounding (ours) | ${(b.rate*100).toFixed(1)}% | ${b.conflicts} | merged via YOLO detections |`);
  lines.push('');
  lines.push('Notes: Heuristic compliance from candidate text. Replace with model-scored file for final numbers.');
  await fsp.mkdir(path.dirname(OUT), { recursive: true });
  await fsp.writeFile(OUT, lines.join('\n'), 'utf-8');
  console.log('Wrote comparison to', OUT);
}

main().catch(e=>{ console.error(e); process.exit(1); });
