#!/usr/bin/env node
/**
 * Rerank inspirations with a family-specific compliance score produced by
 * apply_family_compliance.py. This script reorders candidates within each
 * inspiration and optionally filters to top-k, emitting an updated JSON.
 *
 * Usage:
 *   node rerank_with_compliance.mjs \
 *     --in ../data/inspirations_survey10_scored_flower_vs_cherry.json \
 *     --out ../data/inspirations_survey10_reranked.json \
 *     --family flower_vs_cherry \
 *     --positive flower \
 *     --negative cherry_fruit \
 *     [--top-k 6] [--weight 1.0]
 */

import fs from 'node:fs';
import path from 'node:path';


import { arg, readJson, writeJson, urlOf, hasGlass, probOf } from './utils.mjs';
// arg function now imported from utils.mjs
const IN = arg('--in');
const OUT = arg('--out');
const FAMILY = arg('--family', 'flower_vs_cherry');
const POS = arg('--positive', 'flower');
const NEG = arg('--negative', 'cherry_fruit');
const TOPK = parseInt(arg('--top-k', '0'), 10) || 0;
const WEIGHT = parseFloat(arg('--weight', '1.0')) || 1.0;
const USE_DETECTIONS = Boolean(arg('--use-detections', 'false') && arg('--use-detections') !== 'false');
// Back-compat glass-specific flags
const REQUIRE_GLASS = Boolean(arg('--require-glass', 'false') && arg('--require-glass') !== 'false');
const HARD_REQUIRE_GLASS = Boolean(arg('--hard-require-glass', 'false') && arg('--hard-require-glass') !== 'false');
// New generalized subject-object flags
const SUBJECT_CONFIG = arg('--subject-config', '');
const REQUIRE_SUBJECTS = (arg('--require-subjects', '') || '')
  .split(',')
  .map(s=>s.trim())
  .filter(Boolean);
// Optional forbidden subjects to filter or penalize (e.g., 'flower')
const FORBID_SUBJECTS = (arg('--forbid-subjects', '') || '')
  .split(',')
  .map(s=>s.trim())
  .filter(Boolean);
const HARD_FORBID = Boolean(arg('--hard-forbid', 'false') && arg('--hard-forbid') !== 'false');
const FORBID_PENALTY = parseFloat(arg('--forbid-penalty', '0.25')) || 0.25;
const PENALTY = parseFloat(arg('--penalty', '0.25')) || 0.25;
const NO_GLASS_PENALTY = parseFloat(arg('--no-glass-penalty', '0.5')) || 0.5;
// Graph-based neighbor smoothing flags (Cheb-GR inspired)
const SMOOTH_GAMMA = Math.max(0, Math.min(1, parseFloat(arg('--graph-smooth', '0')) || 0));
const NEIGHBOR_MODE = String(arg('--neighbor-mode', 'detections')); // 'detections' | 'text' | 'both'
const MIN_OVERLAP = parseInt(arg('--neighbor-min-overlap', '1'), 10) || 1;

if (!IN || !OUT) { console.error('Usage: node rerank_with_compliance.mjs --in <scored.json> --out <out.json> [--family F] [--positive flower] [--negative cherry_fruit] [--top-k 6] [--weight 1.0]'); process.exit(1); }

// readJson function now imported from utils.mjs
// writeJson function now imported from utils.mjs); fs.writeFileSync(p, JSON.stringify(obj, null, 2)); }

function probOf(cand, family, cls){
  const fam = (cand.compliance && cand.compliance[family]) || null;
  if (!fam || !fam.probs) return null;
  const p = fam.probs[cls];
  return typeof p === 'number' ? p : null;
}

function baseScore(c){
  // Preserve any upstream score if present; otherwise 0
  return typeof c.score === 'number' ? c.score : 0;
}

function classifyIntent(garnish){
  const s = String(garnish || '').toLowerCase();
  return {
    isFlower: /(blossom|flower|petal|sakura)/.test(s) || /cherry\s+blossom/.test(s),
    isCherryFruit: /\bcherry\b/.test(s) && !(/(blossom|flower|petal|sakura)/.test(s)),
  };
}

function textSignals(txt){
  const t = String(txt || '').toLowerCase();
  return {
    blossom: /(blossom|flower|petal|sakura)/.test(t) || /cherry\s+blossom/.test(t),
    cherry: /\bmaraschino\b|cocktail\s+cherry|\bcherry\b/.test(t)
  };
}

function loadSubjectConfig(p){
  if (!p) return null;
  try { return JSON.parse(fs.readFileSync(p, 'utf-8')); } catch{ return null; }
}
function compileRegexList(list){
  if (!Array.isArray(list)) return [];
  return list.map(x => {
    if (!x) return null;
    if (x instanceof RegExp) return x;
    if (typeof x === 'string') return new RegExp(x, 'i');
    if (typeof x === 'object' && x.pattern) return new RegExp(String(x.pattern), String(x.flags||'i'));
    return null;
  }).filter(Boolean);
}
function normalizeSubjects(spec){
  if (!spec || typeof spec !== 'object') return {};
  const out = {};
  for (const [k,v] of Object.entries(spec)){
    const yolo = Array.isArray(v?.yolo) ? v.yolo.map(s=>String(s).toLowerCase()) : [];
    const text = compileRegexList(v?.text || []);
    out[k] = { yolo, text };
  }
  return out;
}
const SUBJECTS = normalizeSubjects(loadSubjectConfig(SUBJECT_CONFIG)) || normalizeSubjects({
  glass: {
    yolo: ['cup','wine glass','wineglass','glass'],
    text: [/\b(cocktail|drink|beverage|martini|coupe|highball|glass|wine\s+glass)\b/i]
  },
  plate: { yolo: ['plate'], text: [/\bplate\b/i] },
  bowl: { yolo: ['bowl'], text: [/\bbowl\b/i] },
  person: { yolo: ['person'], text: [/\b(person|bartender|hand)\b/i] }
});

function hasDetectionForSubject(cand, subjectKey){
  const det = cand.detected || [];
  if (!Array.isArray(det)) return false;
  const subj = SUBJECTS[subjectKey];
  const accepted = new Set((subj?.yolo || []).map(s=>String(s).toLowerCase()));
  return det.some(d => accepted.has(String(d.name || '').toLowerCase()));
}

function textMatchesSubject(cand, subjectKey){
  const txt = String((cand.alt_description || '') + ' ' + (cand.description || ''));
  const subj = SUBJECTS[subjectKey];
  const regs = subj?.text || [];
  return regs.some(r => r.test(txt));
}

function subjectPresent(cand, requiredSubjects){
  // A candidate is OK if it satisfies ALL required subject keys (AND semantics)
  return requiredSubjects.every(key => hasDetectionForSubject(cand, key) || textMatchesSubject(cand, key));
}

function subjectForbidden(cand, forbiddenSubjects){
  if (!forbiddenSubjects || forbiddenSubjects.length === 0) return false;
  return forbiddenSubjects.some(key => hasDetectionForSubject(cand, key) || textMatchesSubject(cand, key));
}

function detNames(c){
  const det = Array.isArray(c.detected) ? c.detected : [];
  return new Set(det.map(d => String(d.name || '').toLowerCase()).filter(Boolean));
}

function textTokens(c){
  const txt = String((c.alt_description || '') + ' ' + (c.description || '')).toLowerCase();
  return new Set(txt.split(/[^a-z0-9]+/i).filter(w => w && w.length >= 4));
}

function neighborOverlap(a, b){
  let overlap = 0;
  if (NEIGHBOR_MODE === 'detections' || NEIGHBOR_MODE === 'both'){
    const A = detNames(a), B = detNames(b);
    for (const x of A) if (B.has(x)) overlap++;
  }
  if (NEIGHBOR_MODE === 'text' || NEIGHBOR_MODE === 'both'){
    const A = textTokens(a), B = textTokens(b);
    for (const x of A) if (B.has(x)) overlap++;
  }
  return overlap;
}

function rerank(data){
  // Determine which subjects are required
  const requiredSubjects = [...REQUIRE_SUBJECTS];
  if (REQUIRE_GLASS && !requiredSubjects.includes('glass')) requiredSubjects.push('glass');
  const enforceSubjects = USE_DETECTIONS && (requiredSubjects.length > 0);
  for (const rec of data){
    for (const insp of (rec.inspirations || [])){
      let cands = insp.candidates || [];
      // Optional generalized subject-object constraint
      let applyNoSubjectPenalty = false;
      if (enforceSubjects){
        const filtered = cands.filter(c => subjectPresent(c, requiredSubjects));
        if (filtered.length) {
          cands = filtered;
        } else {
          // No candidates satisfy subjects; keep list but penalize in scoring
          applyNoSubjectPenalty = true;
        }
      }
      // Optional forbidden subjects handling
      if (FORBID_SUBJECTS.length > 0){
        if (HARD_FORBID){
          cands = cands.filter(c => !subjectForbidden(c, FORBID_SUBJECTS));
        }
        // If not hard, we'll apply penalties during scoring below
      }
      const intent = classifyIntent(insp?.cocktail?.visual?.garnish);
      let scored = cands.map(c => {
        const ppos = probOf(c, FAMILY, POS) ?? 0;
        const pneg = probOf(c, FAMILY, NEG) ?? 0;
        const compliance = (ppos - pneg); // margin
        let s = baseScore(c) + WEIGHT * compliance;
        // Optional region-aware conflict penalty: when subject is present and text cues conflict with intent (legacy for flower/cherry)
        if (USE_DETECTIONS && (hasDetectionForSubject(c, 'glass'))){
          const txt = (c.alt_description || '') + ' ' + (c.description || '');
          const sig = textSignals(txt);
          const conflictFlower = intent.isFlower && sig.cherry && !sig.blossom;
          const conflictCherry = intent.isCherryFruit && sig.blossom;
          if (conflictFlower || conflictCherry) s -= PENALTY;
        }
        // If we required subjects but none were met, penalize
        if (enforceSubjects && applyNoSubjectPenalty) s -= NO_GLASS_PENALTY;
        // Apply forbidden subject penalty if configured (and not hard-filtered)
        if (FORBID_SUBJECTS.length > 0 && subjectForbidden(c, FORBID_SUBJECTS)) s -= FORBID_PENALTY;
        return { c, s, compliance };
      });

      // Optional graph-based smoothing of scores within inspiration (Cheb-GR inspired)
      if (SMOOTH_GAMMA > 0 && scored.length > 1){
        const newScores = new Array(scored.length).fill(0);
        for (let i=0;i<scored.length;i++){
          const si = scored[i].s;
          let sum = 0, cnt = 0;
          for (let j=0;j<scored.length;j++){
            if (i === j) continue;
            const ov = neighborOverlap(scored[i].c, scored[j].c);
            if (ov >= MIN_OVERLAP){ sum += scored[j].s; cnt++; }
          }
          const nb = cnt > 0 ? (sum / cnt) : si;
          newScores[i] = (1 - SMOOTH_GAMMA) * si + (SMOOTH_GAMMA) * nb;
        }
        for (let i=0;i<scored.length;i++) scored[i].s = newScores[i];
      }

      scored = scored.sort((a,b) => b.s - a.s);
      let newCands = scored.map(x => ({ ...x.c, compliance_margin: x.compliance, rerank_score: x.s }));
      if (TOPK > 0) newCands = newCands.slice(0, TOPK);
      insp.candidates = newCands;
    }
  }
  return data;
}

const data = readJson(IN);
const out = rerank(data);
writeJson(OUT, out);
console.log('Wrote reranked inspirations to', OUT);
