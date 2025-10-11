#!/usr/bin/env node
/**
 * Compare two compliance-scored (and possibly reranked) inspirations files and emit a Markdown report.
 * This is similar to run_dual_ablation_report.mjs but uses the provided files directly (not heuristic-only).
 *
 * Usage:
 *   node run_dual_scored_ablation.mjs \
 *     --a api/image_service/data/inspirations_from_precompute_reranked_flower_vs_cherry.json \
 *     --b api/image_service/data/inspirations_from_yolo_reranked_flower_vs_cherry.json \
 *     --out api/image_service/data/reports/ablation_dual_scored.md \
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

const A = arg('--a');
const B = arg('--b');
const OUT = arg('--out', `api/image_service/data/reports/ablation_${new Date().toISOString().slice(0,10)}_dual_scored.md`);
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
  if (!A || !B){ console.error('Usage: node run_dual_scored_ablation.mjs --a <precompute_reranked.json> --b <yolo_reranked.json> [--out <md>] [--top-k 1]'); process.exit(1); }
  const a = readJson(A);
  const b = readJson(B);
  const ar = aggregate(a, TOPK);
  const br = aggregate(b, TOPK);
  const lines = [];
  lines.push('# Dual Ablation Report (Model-Scored)');
  lines.push('');
  lines.push(`Top-K: ${TOPK}`);
  lines.push('');
  lines.push('| Variant | Compliance@K | Conflicts | Notes |');
  lines.push('|---|---:|---:|---|');
  lines.push(`| Baseline+Compliance | ${(ar.rate*100).toFixed(1)}% | ${ar.conflicts} | precompute_reranked_flower_vs_cherry.json |`);
  lines.push(`| YOLO+Compliance | ${(br.rate*100).toFixed(1)}% | ${br.conflicts} | yolo_reranked_flower_vs_cherry.json |`);
  lines.push('');
  lines.push('Notes: Evaluation uses the same heuristic text-cue compliance, but both inputs are model-reranked.');
  await fsp.mkdir(path.dirname(OUT), { recursive: true });
  await fsp.writeFile(OUT, lines.join('\n'), 'utf-8');
  console.log('Wrote dual scored report to', OUT);
}

main().catch(e=>{ console.error(e); process.exit(1); });
