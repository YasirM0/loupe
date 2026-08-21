// Re-run this whenever a new embedding/NLI model option is added to
// EMBED_MODELS/NLI_MODELS in src/Loupe.jsx, and copy the real percentages
// into that file's `quality` fields — never hand-write a number there.
//
//   node bench/run.mjs
//
// Uses classifyPair/verdictFromScores from src/lib/nli.js — the exact same
// logic the app runs — so results reflect production behavior, not an
// approximation of it. device:'cpu' is deliberate: transformers.js's Node
// build only supports cuda/webgpu/cpu (no 'wasm', unlike the browser build
// the worker uses); cpu is the only one guaranteed to run everywhere.
import { pipeline, env } from '@huggingface/transformers';
import { classifyPair, verdictFromScores } from '../src/lib/nli.js';
import { NLI_TESTSET, RETRIEVAL_TESTSET } from './testset.mjs';

env.allowLocalModels = false;

function cosineSimilarity(a, b) {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
  const denom = Math.sqrt(na) * Math.sqrt(nb);
  return denom ? dot / denom : 0;
}

async function benchNli(modelId) {
  console.log(`\n=== NLI model: ${modelId} ===`);
  const classifier = await pipeline('zero-shot-classification', modelId, { device: 'cpu' });
  let correct = 0;
  for (const item of NLI_TESTSET) {
    const scores = await classifyPair(classifier, item.evidence, item.claim);
    const { status } = verdictFromScores(scores, 1.0);
    const ok = status === item.expected;
    if (ok) correct++;
    console.log(`  ${ok ? 'OK  ' : 'MISS'} expected=${item.expected} got=${status}  "${item.claim.slice(0, 60)}"`);
  }
  const pct = Math.round((correct / NLI_TESTSET.length) * 100);
  console.log(`  -> ${correct}/${NLI_TESTSET.length} = ${pct}%`);
  return pct;
}

async function benchEmbedding(modelId) {
  console.log(`\n=== Embedding model: ${modelId} ===`);
  const embedder = await pipeline('feature-extraction', modelId, { device: 'cpu' });
  const embed = async text => Array.from((await embedder(text, { pooling: 'mean', normalize: true })).data);

  let correct = 0;
  for (const item of RETRIEVAL_TESTSET) {
    const qEmb = await embed(item.claim);
    const poolEmb = [];
    for (const s of item.pool) poolEmb.push(await embed(s));
    const sims = poolEmb.map(e => cosineSimilarity(qEmb, e));
    const bestIdx = sims.indexOf(Math.max(...sims));
    const ok = bestIdx === item.correctIndex;
    if (ok) correct++;
    console.log(`  ${ok ? 'OK  ' : 'MISS'} top-ranked idx=${bestIdx} expected=${item.correctIndex}  "${item.claim.slice(0, 60)}"`);
  }
  const pct = Math.round((correct / RETRIEVAL_TESTSET.length) * 100);
  console.log(`  -> ${correct}/${RETRIEVAL_TESTSET.length} = ${pct}%`);
  return pct;
}

const results = { nli: {}, embedding: {} };

results.nli['Xenova/nli-deberta-v3-base'] = await benchNli('Xenova/nli-deberta-v3-base');
results.nli['Xenova/nli-deberta-v3-small'] = await benchNli('Xenova/nli-deberta-v3-small');

results.embedding['Xenova/bge-base-en-v1.5'] = await benchEmbedding('Xenova/bge-base-en-v1.5');
results.embedding['Xenova/bge-small-en-v1.5'] = await benchEmbedding('Xenova/bge-small-en-v1.5');
results.embedding['Xenova/all-MiniLM-L6-v2'] = await benchEmbedding('Xenova/all-MiniLM-L6-v2');

console.log('\n\n=== FINAL RESULTS (JSON) ===');
console.log(JSON.stringify(results, null, 2));
