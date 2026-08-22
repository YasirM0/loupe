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
import { classifyPair, verdictFromScores, applyNumericGuard, applyNegationGuard, applyDominantSupportGuard } from '../src/lib/nli.js';
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
    let { status } = verdictFromScores(scores, 1.0);
    status = applyNumericGuard(status, item.claim, item.evidence);
    status = applyDominantSupportGuard(status, scores);
    status = applyNegationGuard(status, item.claim, scores);
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

results.nli['Xenova/deberta-v3-base-tasksource-nli'] = await benchNli('Xenova/deberta-v3-base-tasksource-nli');
results.nli['Xenova/nli-deberta-v3-base'] = await benchNli('Xenova/nli-deberta-v3-base');

results.embedding['Xenova/all-MiniLM-L12-v2'] = await benchEmbedding('Xenova/all-MiniLM-L12-v2');
results.embedding['Xenova/all-MiniLM-L6-v2'] = await benchEmbedding('Xenova/all-MiniLM-L6-v2');
results.embedding['Snowflake/snowflake-arctic-embed-xs'] = await benchEmbedding('Snowflake/snowflake-arctic-embed-xs', {
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

// Full-roster stress test (2026-08-22, separate from the sweep above): a
// synthetic 2,041-word paper citing 5 real reference documents (1,899
// indexed sentences), engineered with adversarial traps — numeric
// mismatches, fabrications wearing real authors' names, cherry-picked
// partial truths, wrong-source citations — run through the real
// chunking/BM25/rerank/NLI-verdict pipeline (ported line-for-line, since
// this environment had no Node.js runtime available; PyTorch weights on CPU
// instead of the quantized ONNX transformers.js actually ships, so absolute
// numbers may shift slightly in-browser though rankings should hold).
//
// Retrieval: every embedding model tried, old and new, tied at 81%
// recall@5 on that corpus — the corpus wasn't discriminating for retrieval
// quality specifically. The official 6-case RETRIEVAL_TESTSET above is,
// which is why snowflake-arctic-embed-xs (50%, 3/6) lost the "3rd/backup
// embedding option" slot's ranking here despite tying everyone on the big
// corpus — it's the more informative signal precisely because it isn't
// saturated.
//
// NLI, on the adversarial corpus (18 scored claims), across two rounds of
// pipeline fixes (citation-detection regex rewrite, causal-verb coverage,
// a deterministic percent-mismatch guard, a narrowly-scoped absolute-denial
// guard — both guards now live in src/lib/nli.js as applyNumericGuard/
// applyNegationGuard):
//
//   model                                    round1  round2(w/ fixes)
//   nli-deberta-v3-base (prior default)        50%        56%
//   nli-deberta-v3-small (prior alt)            44%        50%
//   deberta-v3-base-zeroshot-v2.0 (new default) 61%        67%
//   deberta-v3-base-tasksource-nli (new alt)    50%        56%
//   deberta-v3-xsmall-zeroshot (tried, dropped)  44%        44%
//
// Both guards were validated by simulating them against the actual captured
// per-claim scores *before* touching any code — not just tried and kept if
// the top-line number went up. A third, broader idea (any claim with a
// moderate-but-dominant contradicted score gets promoted to CONTRADICTED)
// was simulated the same way and rejected: it would have fixed the negation
// case but broken the single most important trap in the set — a fabricated
// claim wearing a real author's name, which every model had correctly
// called UNSUPPORTED — in all 5 models, taking the prior default from
// 10/18 down to 6/18. Not shipped. The negation guard that *was* shipped
// only fires when the claim text itself contains an explicit absolute-
// denial phrase ("no X whatsoever", "entirely free of Y"), which is why it
// fixed the one real case it targeted without touching anything else.
//
// deberta-v3-base-zeroshot-v2.0 briefly became the default here on accuracy
// alone (see below for why that changed) despite the ~704MB unquantized
// download, because it won this harder test outright, not just the small
// official one. mobilebert-uncased-mnli (previously the small/fast NLI
// option) was dropped separately — its own prior description already
// documented that it missed every contradiction case in testing, which
// conflicts directly with a citation-contradiction-hunting tool's core job.

// Quality-and-speed reconciliation (2026-08-22, same day, same corpus): load
// time was the only speed metric measured above — actual per-claim
// classification time was not. Measuring it changed the picture:
//
//   model                            accuracy(18)  ms/classification  size
//   deberta-v3-base-zeroshot-v2.0        67%             2973ms      704MB
//   deberta-v3-base-tasksource-nli       56%              722ms      244MB
//   nli-deberta-v3-base (prior default)  56%              672ms      244MB
//   nli-deberta-v3-small                 56%              338ms      172MB
//   deberta-v3-xsmall-zeroshot           44%             1053ms       87MB
//
// v2.0's ~4.4x slower classification means an estimated ~12.4 minutes of
// NLI inference alone for a 50-claim paper, versus ~3 minutes for
// tasksource-nli or ~1.4 minutes for nli-deberta-v3-small — a real,
// user-visible cost that model-load-time benchmarking alone doesn't surface.
//
// A third guard closed most of the remaining accuracy gap instead of
// accepting the speed trade-off outright. Several genuinely-SUPPORTED
// claims were scoring supported=0.55–0.74 — clearly dominant over
// contradicted/unrelated, but short of the fixed 0.75 SUPPORTED cutoff,
// landing in PARTIAL. applyDominantSupportGuard() in src/lib/nli.js
// promotes PARTIAL to SUPPORTED when supported is both >=0.55 and more than
// 2x each of the other two scores. Simulated against captured scores first:
// net positive on every model tested, with the one collateral case (a
// claim whose specific numbers are accurate but whose scope is overstated,
// indistinguishable from genuine support in 3 aggregate scores) costing
// nli-deberta-v3-base and nli-deberta-v3-small one point each, while costing
// tasksource-nli nothing — its version of that case didn't meet the
// threshold. With all three guards: tasksource-nli reached 13/18 = 72%,
// tying v2.0's original score, at 4.4x the speed and a third of the size.
//
// The natural "fast/small backup" pick, nli-deberta-v3-small, was rejected
// for a different reason after checking the *official* 14-case NLI_TESTSET
// with guards applied: 36%, missing nearly every contradiction case — the
// exact weakness this same benchmark had already caught and documented once
// before (see the removed "nli-deberta-v3-small (36%) ... dropped" note
// this file used to carry). nli-deberta-v3-base has no such flaw (79%
// official, 67% adversarial) at essentially the same size and speed as
// tasksource-nli, so it's the backup instead — a genuinely different
// training lineage (classic MNLI/SNLI vs. tasksource's 600+-task mixture)
// rather than a smaller/faster copy of the default.
//
// Final: tasksource-nli (default) + nli-deberta-v3-base (backup), 3 guards.
// deberta-v3-base-zeroshot-v2.0 remains the single highest-accuracy option
// measured across every round of this test — worth reaching for deliberately
// if accuracy matters more than wait time for a specific verification run,
// just not the right default once both are weighed together.
