#!/usr/bin/env node
/**
 * CoTRR-lite listwise reranker with subject–object constraints.
 *
 * Training-free, uses:
 *  - Compliance margin (CLIP-based head already computed)
 *  - Subject/object constraints (require-glass, detection/text fallback)
 *  - Optional LLM listwise re-ranking via local proxy (/api/generate)
 *
 * Inputs: inspirations JSON with candidates and compliance scores.
 * Output: same shape with candidates re-ordered and annotated with rerank_score.
 *
 * Example:
 *   node rerank_listwise_llm.mjs \
 *     --in api/image_service/data/inspirations_from_precompute_scored_flower_vs_cherry.json \
 *     --out api/image_service/data/inspirations_from_precompute_reranked_cotrr_lite.json \
 *     --family flower_vs_cherry --positive flower --negative cherry_fruit \
 *     --top-k 6 --weight 1.0 --llm-weight 0.6 \
 *     --use-detections true --require-glass true \
 *     --proxy-url http://127.0.0.1:8787/api/generate
 */

import fs from 'node:fs';
import path from 'node:path';

function arg(name, defVal = null){ const i = process.argv.indexOf(name); return i === -1 ? defVal : (process.argv[i+1] ?? true); }
const IN = arg('--in');
const OUT = arg('--out');
const FAMILY = arg('--family', 'flower_vs_cherry');
const POS = arg('--positive', 'flower');
const NEG = arg('--negative', 'cherry_fruit');
const TOPK = parseInt(arg('--top-k', '0'), 10) || 0;
const WEIGHT = parseFloat(arg('--weight', '1.0')) || 1.0;          // compliance weight
const LLM_WEIGHT = parseFloat(arg('--llm-weight', '0.6')) || 0.6;   // listwise llm weight
const SIMULATE_LLM = Boolean(arg('--simulate-llm', 'false') && arg('--simulate-llm') !== 'false');
const USE_DETECTIONS = Boolean(arg('--use-detections', 'false') && arg('--use-detections') !== 'false');
const REQUIRE_GLASS = Boolean(arg('--require-glass', 'false') && arg('--require-glass') !== 'false');
const HARD_REQUIRE_GLASS = Boolean(arg('--hard-require-glass', 'false') && arg('--hard-require-glass') !== 'false');
const PENALTY = parseFloat(arg('--penalty', '0.25')) || 0.25;
const NO_GLASS_PENALTY = parseFloat(arg('--no-glass-penalty', '0.5')) || 0.5;
const PROXY_URL = (() => {
  const cli = arg('--proxy-url', null);
  if (cli) return cli;
  if (process.env.LOCAL_PROXY_URL) return process.env.LOCAL_PROXY_URL;
  try {
    const port = String(fs.readFileSync('.local_api_port','utf-8')).trim();
    const n = parseInt(port, 10);
    if (!Number.isNaN(n)) return `http://127.0.0.1:${n}/api/generate`;
  } catch {}
  return 'http://127.0.0.1:8787/api/generate';
})();
const DRY = Boolean(arg('--dry-run', 'false') && arg('--dry-run') !== 'false');
// Gating options for safer LLM usage
const LLM_TOP_M = parseInt(arg('--llm-top-m', '0'), 10) || 0;         // 0 means no restriction
const LLM_TIE_EPS = parseFloat(arg('--llm-tie-eps', '0')) || 0;       // 0 means disabled

if (!IN || !OUT) { console.error('Usage: node rerank_listwise_llm.mjs --in <scored.json> --out <out.json> [--family F] [--positive flower] [--negative cherry_fruit] [--top-k 6] [--weight 1.0] [--llm-weight 0.6] [--proxy-url http://127.0.0.1:8787/api/generate]'); process.exit(1); }

function readJson(p){ return JSON.parse(fs.readFileSync(p, 'utf-8')); }
function writeJson(p,obj){ fs.mkdirSync(path.dirname(p), { recursive: true }); fs.writeFileSync(p, JSON.stringify(obj, null, 2)); }

function probOf(cand, family, cls){ const fam = (cand.compliance && cand.compliance[family]) || null; if (!fam || !fam.probs) return null; const v = fam.probs[cls]; return typeof v === 'number' ? v : null; }
function baseScore(c){ return typeof c.score === 'number' ? c.score : 0; }
function hasGlassDetection(cand){ const det = cand.detected || []; if (!Array.isArray(det)) return false; const GLASS = new Set(['cup','wine glass','wineglass','glass']); return det.some(d => GLASS.has(String(d.name || '').toLowerCase())); }
function textHasGlass(cand){ const txt = String((cand.alt_description || '') + ' ' + (cand.description || '')).toLowerCase(); return /(cocktail|drink|beverage|martini|coupe|highball|glass|wine\s+glass)/.test(txt); }
function classifyIntent(garnish){ const s = String(garnish || '').toLowerCase(); return { isFlower: /(blossom|flower|petal|sakura)/.test(s) || /cherry\s+blossom/.test(s), isCherryFruit: /\bcherry\b/.test(s) && !(/(blossom|flower|petal|sakura)/.test(s)) }; }
function textSignals(txt){ const t = String(txt || '').toLowerCase(); return { blossom: /(blossom|flower|petal|sakura)/.test(t) || /cherry\s+blossom/.test(t), cherry: /\bmaraschino\b|cocktail\s+cherry|\bcherry\b/.test(t) }; }

async function callLLMListwise(query, candidates){
  // Build a compact, privacy-safe prompt using local metadata only.
  const parts = [];
  parts.push('Task: Rank images for a cocktail retrieval query using subject–object constraints.');
  parts.push('Query: ' + (query || ''));
  parts.push('Rules: Glass must be present (subject). Prefer garnish that matches intent. Penalize mismatches (flower vs cherry fruit).');
  parts.push('Candidates: Provide a sorted list of candidate indexes (0-based) best to worst, with a short reason.');
  const lines = candidates.map((c, idx) => {
    const txt = [c.alt_description, c.description].filter(Boolean).join(' ').slice(0, 180);
    const det = Array.isArray(c.detected) ? c.detected.map(d=>d.name).slice(0,6) : [];
    const fam = c.compliance && c.compliance[FAMILY];
    const pm = typeof c.compliance_margin === 'number' ? c.compliance_margin : (fam ? ((fam.probs?.[POS]||0)-(fam.probs?.[NEG]||0)) : 0);
    return `${idx}. id=${c.id || c.slug || c.url || idx}; score=${c.score ?? 0}; cm=${pm.toFixed(3)}; det=[${det.join(', ')}]; text="${txt}"`;
  });
  parts.push(lines.join('\n'));
  parts.push('Respond ONLY JSON with no extra prose: { "order": [indexes...], "notes": "..." }');

  const prompt = parts.join('\n');
  try {
    const r = await fetch(PROXY_URL, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ prompt }) });
    const js = await r.json().catch(()=>null);
    // Our proxy normalizes to { cocktails: [...] } but we sent general prompt; handle flexible shapes.
    let text = null;
    if (js && Array.isArray(js.cocktails)) {
      // If model returned an array, try to find an object with order
      const found = js.cocktails.find(x => x && (x.order || x.rank || x.sorted || x.indexes));
      if (found && (found.order || found.indexes || found.rank || found.sorted)) {
        return found;
      }
      text = js.cocktails[0];
    } else if (js && js.order) {
      return js;
    } else if (js && js.error && js.assistantText) {
      text = js.assistantText;
    } else {
      text = JSON.stringify(js);
    }
    // Try to parse JSON inside text
    if (typeof text === 'string'){
      const m = text.match(/\{[\s\S]*\}/m);
      if (m){ try { return JSON.parse(m[0]); } catch {} }
    }
  } catch (e) {
    return null;
  }
  return null;
}

function linearBlend(base, llmScore, w){ return (1-w)*base + w*llmScore; }

async function rerankRecord(rec){
  for (const insp of (rec.inspirations || [])){
    let cands = insp.candidates || [];
    let applyNoGlassPenalty = false;
    if (USE_DETECTIONS && (REQUIRE_GLASS || HARD_REQUIRE_GLASS)){
      const filtered = cands.filter(c => hasGlassDetection(c) || textHasGlass(c));
      if (filtered.length) cands = filtered; else applyNoGlassPenalty = true;
    }
    const intent = classifyIntent(insp?.cocktail?.visual?.garnish);
    // Base score with compliance and penalties
    let scored = cands.map((c, idx) => {
      const ppos = probOf(c, FAMILY, POS) ?? 0; const pneg = probOf(c, FAMILY, NEG) ?? 0; const compliance = (ppos - pneg);
      let s = baseScore(c) + WEIGHT * compliance;
      // Apply conflict penalty if glass is evident via detection OR text cues (broadened)
      if (USE_DETECTIONS && (hasGlassDetection(c) || textHasGlass(c))){
        const txt = (c.alt_description || '') + ' ' + (c.description || '');
        const sig = textSignals(txt);
        const conflictFlower = intent.isFlower && sig.cherry && !sig.blossom;
        const conflictCherry = intent.isCherryFruit && sig.blossom;
        if (conflictFlower || conflictCherry) s -= PENALTY;
      }
      if (USE_DETECTIONS && REQUIRE_GLASS && applyNoGlassPenalty){
        const hasAny = hasGlassDetection(c) || textHasGlass(c);
        if (!hasAny) s -= NO_GLASS_PENALTY;
      }
      return { idx, c, base: s, compliance };
    });

    // Optional LLM listwise rerank
    let order = null;
    let llmNotes = null;
  if (!DRY && scored.length >= 2){
      if (SIMULATE_LLM){
        // Deterministic heuristic to simulate listwise ordering using text cues
        const intent = classifyIntent(insp?.cocktail?.visual?.garnish);
        const arr = cands.map((c, idx) => {
          const txt = (c.alt_description || '') + ' ' + (c.description || '');
          const sig = textSignals(txt);
          const glassBonus = (hasGlassDetection(c) || textHasGlass(c)) ? 0.25 : -0.25;
          let good=0,bad=0;
          if (intent.isFlower){ if (sig.blossom) good++; if (sig.cherry && !sig.blossom) bad++; }
          if (intent.isCherryFruit){ if (sig.cherry) good++; if (sig.blossom) bad++; }
          const h = good - bad + glassBonus;
          return { idx, h, good, bad, glassBonus };
        }).sort((a,b)=> b.h - a.h || a.idx - b.idx);
        order = arr.map(x=>x.idx);
        llmNotes = 'simulated-order-by-text-signals';
      } else {
        const query = insp.tokenQuery || insp.query || insp.title || rec?.tokenQuery || rec?.title || '';
        const llm = await callLLMListwise(query, cands);
        if (llm && Array.isArray(llm.order)){
          // Normalize to unique indices within range
          const set = new Set();
          order = llm.order.map(x => Number(x)).filter(n => Number.isInteger(n) && n>=0 && n<cands.length && !set.has(n) && set.add(n));
          llmNotes = llm.notes || null;
          // If LLM provided ids, try to map to indexes
          if (order.length === 0 && Array.isArray(llm.order)){
            const idToIdx = new Map(cands.map((c,i)=>[c.id||c.slug||c.url, i]));
            for (const any of llm.order){ const i = idToIdx.get(any); if (typeof i === 'number' && !set.has(i)){ order.push(i); set.add(i); } }
          }
        }
      }
    }

    // If simulation requested but no order produced, generate a safe fallback order
    if (SIMULATE_LLM && (!order || order.length === 0) && cands.length > 0){
      order = Array.from({length: cands.length}, (_,i)=>i);
      llmNotes = (llmNotes ? llmNotes + '; ' : '') + 'sim-fallback-order';
    }

    // Blend scores with gating
    if (order && order.length){
      const llmRank = new Map(order.map((idx, rank) => [idx, rank]));
      const maxRank = Math.max(...llmRank.values());
      // Precompute base-order and neighbor min diffs
      const baseSorted = [...scored].sort((a,b)=> b.base - a.base);
      const baseRank = new Map(baseSorted.map((x,i)=>[x.idx, i]));
      const neighborMinDiff = new Map();
      for (let i=0;i<baseSorted.length;i++){
        const prev = i>0 ? Math.abs(baseSorted[i].base - baseSorted[i-1].base) : Infinity;
        const next = i<baseSorted.length-1 ? Math.abs(baseSorted[i].base - baseSorted[i+1].base) : Infinity;
        neighborMinDiff.set(baseSorted[i].idx, Math.min(prev, next));
      }
      scored = scored.map(x => {
        const r = llmRank.has(x.idx) ? llmRank.get(x.idx) : maxRank + 1;
        const llmScore = 1 - (r / (maxRank + 1)); // higher is better
        const br = baseRank.get(x.idx) ?? 0;
        const nearTie = LLM_TIE_EPS > 0 ? (neighborMinDiff.get(x.idx) <= LLM_TIE_EPS) : true;
        const withinTop = LLM_TOP_M > 0 ? (br < LLM_TOP_M) : true;
        const gate = nearTie && withinTop;
        const wEff = gate ? LLM_WEIGHT : 0;
        const s = linearBlend(x.base, llmScore, wEff);
        return { ...x, s, llmScore, llmRank: r, llmGate: gate };
      }).sort((a,b)=> b.s - a.s);
    } else {
      scored = scored.map(x => ({ ...x, s: x.base, llmRank: null, llmScore: null, llmGate: false })).sort((a,b)=> b.s - a.s);
    }

  let newCands = scored.map(x => ({ ...x.c, compliance_margin: x.compliance, rerank_score: x.s, llm_rank: x.llmRank, llm_score: x.llmScore, llm_gate: x.llmGate }));
    if (TOPK > 0) newCands = newCands.slice(0, TOPK);
    insp.candidates = newCands;
    // annotate inspiration-level LLM usage for analysis
    insp.llm_used = Boolean(order && order.length);
    if (insp.llm_used) insp.llm_order = order;
    if (llmNotes) insp.llm_notes = String(llmNotes).slice(0, 400);
  }
  return rec;
}

async function main(){
  const data = readJson(IN);
  const out = Array.isArray(data) ? [] : null;
  if (!Array.isArray(data)) {
    // single record shape
    const rec = await rerankRecord(data);
    writeJson(OUT, rec);
    console.log('Wrote CoTRR-lite reranked inspirations to', OUT);
    return;
  }
  for (const rec of data){ out.push(await rerankRecord(rec)); }
  writeJson(OUT, out);
  console.log('Wrote CoTRR-lite reranked inspirations to', OUT);
}

main().catch(e => { console.error(e); process.exit(1); });
