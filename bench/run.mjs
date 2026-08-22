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

// query()/passage() let a model apply its documented prefix convention
// (e.g. intfloat/e5-* requires "query: "/"passage: " prefixes to perform as
// documented — benchmarking it unprefixed would understate it unfairly).
// Defaults to no prefix for models that don't need one.
async function benchEmbedding(modelId, { query = t => t, passage = t => t, pooling = 'mean' } = {}) {
  console.log(`\n=== Embedding model: ${modelId} ===`);
  const loadStart = performance.now();
  const embedder = await pipeline('feature-extraction', modelId, { device: 'cpu' });
  const loadMs = Math.round(performance.now() - loadStart);
  const embed = async text => Array.from((await embedder(text, { pooling, normalize: true })).data);

  let correct = 0;
  let embedMs = 0;
  for (const item of RETRIEVAL_TESTSET) {
    let t0 = performance.now();
    const qEmb = await embed(query(item.claim));
    embedMs += performance.now() - t0;
    const poolEmb = [];
    for (const s of item.pool) {
      t0 = performance.now();
      poolEmb.push(await embed(passage(s)));
      embedMs += performance.now() - t0;
    }
    const sims = poolEmb.map(e => cosineSimilarity(qEmb, e));
    const bestIdx = sims.indexOf(Math.max(...sims));
    const ok = bestIdx === item.correctIndex;
    if (ok) correct++;
    console.log(`  ${ok ? 'OK  ' : 'MISS'} top-ranked idx=${bestIdx} expected=${item.correctIndex}  "${item.claim.slice(0, 60)}"`);
  }
  const pct = Math.round((correct / RETRIEVAL_TESTSET.length) * 100);
  const totalEmbeds = RETRIEVAL_TESTSET.length + RETRIEVAL_TESTSET.reduce((n, i) => n + i.pool.length, 0);
  const avgMsPerEmbed = Math.round(embedMs / totalEmbeds);
  console.log(`  -> ${correct}/${RETRIEVAL_TESTSET.length} = ${pct}%  |  load ${loadMs}ms  |  avg ${avgMsPerEmbed}ms/embed (cpu, this machine — relative comparison only)`);
  return { pct, loadMs, avgMsPerEmbed };
}

const results = { nli: {}, embedding: {} };

results.nli['Xenova/nli-deberta-v3-base'] = await benchNli('Xenova/nli-deberta-v3-base');
results.nli['Xenova/nli-deberta-v3-small'] = await benchNli('Xenova/nli-deberta-v3-small');

results.embedding['Xenova/all-MiniLM-L6-v2'] = await benchEmbedding('Xenova/all-MiniLM-L6-v2');
results.embedding['Xenova/bge-small-en-v1.5'] = await benchEmbedding('Xenova/bge-small-en-v1.5');
results.embedding['Xenova/e5-small-v2'] = await benchEmbedding('Xenova/e5-small-v2', {
  query: t => `query: ${t}`, passage: t => `passage: ${t}`,
});
results.embedding['Xenova/all-MiniLM-L12-v2'] = await benchEmbedding('Xenova/all-MiniLM-L12-v2');
results.embedding['Snowflake/snowflake-arctic-embed-s'] = await benchEmbedding('Snowflake/snowflake-arctic-embed-s', {
  query: t => `Represent this sentence for searching relevant passages: ${t}`,
  pooling: 'cls',
});

console.log('\n\n=== FINAL RESULTS (JSON) ===');
console.log(JSON.stringify(results, null, 2));

// One-off wider sweep (2026-08-22), same RETRIEVAL_TESTSET/methodology,
// prompted by a user-run experiment showing bge-base-en-v1.5 scoring higher
// than all-MiniLM-L6-v2 on a real paper despite our 6-case set tying them at
// 100%. Nominated candidates from the MiniLM/E5/BGE-small class (each given
// its documented prefix/pooling convention — e5 needs "query:"/"passage:",
// arctic-embed needs a query prefix + cls pooling, not mean):
//
//   model                              acc(6)  ms/embed(cpu)  quantized size
//   all-MiniLM-L6-v2 (shipped default)  100%       7ms          21.9MB
//   bge-base-en-v1.5 (shipped alt)      100%      29ms         105.0MB
//   bge-small-en-v1.5 (dropped)          83%      12ms          32.4MB
//   gte-small                            67%      12ms          32.4MB
//   e5-small-v2                         100%      12ms          32.4MB
//   all-MiniLM-L12-v2                   100%      12ms          32.4MB
//   jina-embeddings-v2-small-en          83%      10ms          31.2MB
//   snowflake-arctic-embed-s             100%      11ms          32.4MB
//
// Every new candidate that hit 100% is both larger and slower than the
// shipped default, which already wins every axis — so nothing here changed
// EMBED_MODELS. bge-base-en-v1.5 was subsequently removed from EMBED_MODELS
// entirely after real-world use confirmed the same thing this bench already
// showed: no accuracy edge over the smaller/faster options, just slower.
// This also means the *specific* recurring contradiction the
// user hit is not an embedding-model-quality problem: their own real-paper
// test showed the same contradiction from both MiniLM-L6 and bge-base, and
// bge-base ties for best on this synthetic set too. That points at retrieval
// ranking or NLI judgment on whichever evidence got selected, not embedding
// choice — needs the actual paper/claim/sources to diagnose further, which
// this synthetic 6-case set (by design, ceiling-effect-prone) can't resolve.
