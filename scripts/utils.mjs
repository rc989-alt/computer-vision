#!/usr/bin/env node
/**
 * Shared utilities for image_service scripts.
 * Consolidates common functions used across multiple scripts.
 */
import fs from 'node:fs';
import path from 'node:path';

export function arg(name, def = null) {
  const i = process.argv.indexOf(name);
  return i === -1 ? def : (process.argv[i + 1] ?? true);
}

export function argAll(flag) {
  const i = process.argv.indexOf(flag);
  return i === -1 ? [] : process.argv.slice(i + 1).filter(s => !s.startsWith('--'));
}

export function readJson(p) {
  return JSON.parse(fs.readFileSync(p, 'utf-8'));
}

export function writeJson(p, obj) {
  fs.mkdirSync(path.dirname(p), { recursive: true });
  fs.writeFileSync(p, JSON.stringify(obj, null, 2));
}

export function urlOf(c) {
  const u = c?.urls || {};
  return u.regular || u.small || u.full || u.raw || '';
}

export function dateStr() {
  return new Date().toISOString().slice(0, 10).replace(/-/g, '');
}

export function loadSubjectConfig(p) {
  if (!p) return null;
  try {
    return JSON.parse(fs.readFileSync(p, 'utf-8'));
  } catch {
    return null;
  }
}

export function compileRegexList(list) {
  if (!Array.isArray(list)) return [];
  return list.map(x => {
    if (!x) return null;
    if (x instanceof RegExp) return x;
    if (typeof x === 'string') return new RegExp(x, 'i');
    if (typeof x === 'object' && x.pattern) return new RegExp(String(x.pattern), String(x.flags || 'i'));
    return null;
  }).filter(Boolean);
}

export function normalizeSubjects(spec) {
  if (!spec || typeof spec !== 'object') return {};
  const out = {};
  for (const [k, v] of Object.entries(spec)) {
    const yolo = Array.isArray(v?.yolo) ? v.yolo.map(s => String(s).toLowerCase()) : [];
    const text = compileRegexList(v?.text || []);
    out[k] = { yolo, text };
  }
  return out;
}

export function probOf(c, family, cls) {
  const fam = (c.compliance && c.compliance[family]) || null;
  if (!fam || !fam.probs) return null;
  const p = fam.probs[cls];
  return typeof p === 'number' ? p : null;
}

export function hasGlass(c) {
  const det = c.detected || [];
  if (!Array.isArray(det)) return false;
  const GLASS = new Set(['cup', 'wine glass', 'wineglass', 'glass']);
  return det.some(d => GLASS.has(String(d.name || '').toLowerCase()));
}