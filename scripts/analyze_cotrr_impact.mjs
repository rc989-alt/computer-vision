#!/usr/bin/env node
/**
 * Analyze CoTRR-lite impact vs Subject/Object on the same precompute set.
 * Reports: total inspirations, % with LLM used, top-1 change count, and compliance@K deltas.
 *
 * Usage:
 *   node analyze_cotrr_impact.mjs \
 *     --subject api/image_service/data/inspirations_from_precompute_reranked_flower_vs_cherry.json \
 *     --cotrr api/image_service/data/inspirations_from_precompute_reranked_cotrr_lite.json \
 *     --ks 1,3
 */
import fs from 'node:fs';

function arg(name, defVal=null){ const i = process.argv.indexOf(name); return i === -1 ? defVal : (process.argv[i+1] ?? true); }
function readJson(p){ return JSON.parse(fs.readFileSync(p, 'utf-8')); }

const SUBJECT = arg('--subject', 'api/image_service/data/inspirations_from_precompute_reranked_flower_vs_cherry.json');
const COTRR = arg('--cotrr', 'api/image_service/data/inspirations_from_precompute_reranked_cotrr_lite.json');
const KS = String(arg('--ks', '1,3')).split(',').map(s => parseInt(s.trim(), 10)).filter(v => !Number.isNaN(v) && v>0);

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
    lime: /\blime\b/.test(t), orange: /\borange\b/.test(t), lemon: /\blemon\b/.test(t), grapefruit: /\bgrapefruit\b/.test(t)
  };
}
function compliantTopK(insp, k){
  const intent = classifyIntent(insp?.cocktail?.visual?.garnish);
  const cands = (insp?.candidates || []).slice(0, k);
  if (!cands.length) return { ok:false, conflicts:0 };
  let ok=false, conf=0;
  for (const c of cands){
    const txt = (c.alt_description || '') + ' ' + (c.description || '');
    const s = textSignals(txt);
    let good=0,bad=0;
    if (intent.isFlower){ if (s.blossom) good++; if (s.cherry && !s.blossom) bad++; }
    if (intent.isCherryFruit){ if (s.cherry) good++; if (s.blossom) bad++; }
    if (intent.isZest){ if (s.zest) good++; if (s.chunk) bad++; }
    if (intent.isChunk){ if (s.chunk) good++; if (s.zest) bad++; }
    if (intent.fruit){ if (s[intent.fruit]) good++; }
    ok = ok || (good>0);
    if (bad>0) conf++;
  }
  return { ok, conflicts: conf };
}

function aggregate(data, ks){
  const out = {};
  for (const k of ks){ out[k] = { total:0, compliant:0, conflicts:0 }; }
  for (const rec of data||[]){
    for (const insp of (rec.inspirations||[])){
      for (const k of ks){
        const r = compliantTopK(insp,k);
        out[k].total += 1; out[k].compliant += r.ok ? 1 : 0; out[k].conflicts += r.conflicts;
      }
    }
  }
  for (const k of ks){ const o=out[k]; o.rate = o.total? o.compliant/o.total : 0; }
  return out;
}

function top1Id(insp){ const c=(insp?.candidates||[])[0]; return c? (c.id||c.slug||c.url||null) : null; }

function countTop1Changes(a,b){
  let total=0, changed=0;
  for (let i=0;i<Math.min(a.length,b.length);i++){
    const A=a[i].inspirations||[]; const B=b[i].inspirations||[];
    for (let j=0;j<Math.min(A.length,B.length);j++){
      total++;
      if (top1Id(A[j]) !== top1Id(B[j])) changed++;
    }
  }
  return { total, changed, rate: total? changed/total : 0 };
}

function countLLMUsed(b){
  let total=0, used=0;
  for (const rec of b||[]){
    for (const insp of (rec.inspirations||[])){
      total++; if (insp.llm_used) used++;
    }
  }
  return { total, used, rate: total? used/total : 0 };
}

const subj = readJson(SUBJECT);
const cotrr = readJson(COTRR);
const aggA = aggregate(subj, KS);
const aggB = aggregate(cotrr, KS);
const top1 = countTop1Changes(subj, cotrr);
const llm = countLLMUsed(cotrr);

const summary = { ks: KS, subject: aggA, cotrr: aggB, top1Changes: top1, llmUsage: llm };
console.log(JSON.stringify(summary, null, 2));
