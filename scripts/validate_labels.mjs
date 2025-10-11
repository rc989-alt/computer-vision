#!/usr/bin/env node
/**
 * Validate a labels JSON file's schema and basic constraints.
 * Usage:
 *   node api/image_service/scripts/validate_labels.mjs <labels.json>
 */
import fs from 'node:fs';

const path = process.argv[2];
if (!path) { console.error('Provide a labels JSON path'); process.exit(1); }
if (!fs.existsSync(path)) { console.error('File not found:', path); process.exit(1); }

const validFamilies = new Set(['flower_vs_cherry']);
const validLabels = new Set(['flower', 'cherry_fruit', 'unknown']);

let data;
try { data = JSON.parse(fs.readFileSync(path, 'utf-8')); }
catch (e) { console.error('Invalid JSON:', e?.message || e); process.exit(1); }

if (!Array.isArray(data)) { console.error('Expected an array of label rows.'); process.exit(1); }

let ok = true;
const seen = new Set();
for (let i=0; i<data.length; i++){
  const r = data[i];
  const where = `row ${i}`;
  const tokenId = r?.tokenId;
  const inspirationIndex = r?.inspirationIndex;
  const candidateId = r?.candidateId;
  const family = r?.family;
  const label = r?.label;
  if (!tokenId || typeof tokenId !== 'string'){ console.error(where, 'missing tokenId'); ok=false; }
  if (typeof inspirationIndex !== 'number'){ console.error(where, 'missing/invalid inspirationIndex'); ok=false; }
  if (!candidateId || typeof candidateId !== 'string'){ console.error(where, 'missing/invalid candidateId'); ok=false; }
  if (!validFamilies.has(family)){ console.error(where, `invalid family: ${family}`); ok=false; }
  if (!validLabels.has(label)){ console.error(where, `invalid label: ${label}`); ok=false; }
  const key = [tokenId, inspirationIndex, candidateId].join('::');
  if (seen.has(key)) { console.error(where, 'duplicate key:', key); ok=false; }
  seen.add(key);
}

if (!ok){
  console.error('Validation FAILED');
  process.exit(2);
}
console.log('Validation OK:', data.length, 'rows');
