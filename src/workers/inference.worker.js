import { pipeline, env } from '@huggingface/transformers';
import { BM25Index, cosineSimilarity } from '../lib/bm25.js';
import { classifyPairsBatch, verdictFromScores, applyNumericGuard, applyNegationGuard, applyDominantSupportGuard } from '../lib/nli.js';

env.allowLocalModels = false;

let embedder = null;
let embedderName = null;
let classifier = null;
let classifierName = null;

// A model's documented embedding convention — asymmetric models (e5,
// arctic-embed) need queries and reference passages prefixed differently,
// and/or CLS pooling instead of mean, or they score much worse than their
// benchmark numbers suggest. Set alongside the embedder, in Loupe.jsx's
// EMBED_MODELS.
let embedPooling = 'mean';
let embedQueryPrefix = '';
let embedPassagePrefix = '';

let bm25 = null;
let sentenceEmbeddings = null; // parallel array to bm25.docs, or null if not needed

async function ensureModels({ embedModel, nliModel, retrievalMethod, nliDtype, embedPooling: pooling, embedQueryPrefix: queryPrefix, embedPassagePrefix: passagePrefix }) {
  embedPooling = pooling || 'mean';
  embedQueryPrefix = queryPrefix || '';
  embedPassagePrefix = passagePrefix || '';
  if (retrievalMethod !== 'bm25' && embedderName !== embedModel) {
    embedder = await pipeline('feature-extraction', embedModel, {
      device: 'wasm',
      progress_callback: p => self.postMessage({ type: 'MODEL_PROGRESS', model: 'embedding', ...p }),
    });
    embedderName = embedModel;
  }
  if (classifierName !== nliModel) {
    // Most ONNX repos ship several quantized variants and the library
    // defaults to picking a small one — but not every repo has those (e.g.
    // MoritzLaurer's zeroshot model only exports fp32), so a model entry
    // can force a specific dtype via NLI_MODELS[...].dtype in Loupe.jsx.
    classifier = await pipeline('zero-shot-classification', nliModel, {
      device: 'wasm',
      ...(nliDtype ? { dtype: nliDtype } : {}),
      progress_callback: p => self.postMessage({ type: 'MODEL_PROGRESS', model: 'nli', ...p }),
    });
    classifierName = nliModel;
  }
}

// Only ever called with a claim's text (the "query" side of retrieval) —
// see verifyClaims below, its one caller.
async function embed(text) {
  const out = await embedder(embedQueryPrefix + text, { pooling: embedPooling, normalize: true });
  return Array.from(out.data);
}

// Batched calls are meaningfully faster than N one-at-a-time calls (verified:
// identical output, ~2x+ faster even at small batch sizes) since the model
// runs one larger matrix operation instead of many small ones. 16 keeps
// memory bounded for very large reference sets while still getting most of
// the benefit.
const EMBED_BATCH_SIZE = 16;

// Only ever called with reference-sentence text (the "passage" side of
// retrieval) — see buildIndex below, its one caller.
async function embedBatch(texts) {
  const out = await embedder(texts.map(t => embedPassagePrefix + t), { pooling: embedPooling, normalize: true });
  const [n, dim] = out.dims;
  const result = new Array(n);
  for (let i = 0; i < n; i++) result[i] = Array.from(out.data.slice(i * dim, (i + 1) * dim));
  return result;
}

async function buildIndex(sentences, retrievalMethod) {
  bm25 = new BM25Index(sentences);
  sentenceEmbeddings = null;
  if (retrievalMethod !== 'bm25') {
    sentenceEmbeddings = new Array(sentences.length);
    for (let start = 0; start < sentences.length; start += EMBED_BATCH_SIZE) {
      const end = Math.min(start + EMBED_BATCH_SIZE, sentences.length);
      const batchEmbeddings = await embedBatch(sentences.slice(start, end).map(s => s.text));
      for (let j = 0; j < batchEmbeddings.length; j++) sentenceEmbeddings[start + j] = batchEmbeddings[j];
      self.postMessage({ type: 'INDEX_PROGRESS', done: end, total: sentences.length });
    }
  }
}

// Which reference sentences to compare a claim against, per retrieval method.
// Top-5 (not top-3) and a wider BM25 candidate pool — more evidence for the
// NLI step to weigh per claim, trading some speed for closer-to-LLM recall.
const EVIDENCE_TOP_K = 5;
const BM25_POOL = 30;

// claimEmbedding is computed once by the caller and passed in — retrieval
// and the later topCosine calculation both need the same claim embedding,
// and recomputing it twice per claim was pure waste.
async function retrieve(claimText, retrievalMethod, claimEmbedding) {
  if (retrievalMethod === 'bm25') {
    return bm25.search(claimText, EVIDENCE_TOP_K).map(r => r.doc);
  }
  if (retrievalMethod === 'embeddings') {
    const scored = bm25.docs.map((doc, i) => ({ doc, sim: cosineSimilarity(claimEmbedding, sentenceEmbeddings[i]) }));
    scored.sort((a, b) => b.sim - a.sim);
    return scored.slice(0, EVIDENCE_TOP_K).map(s => s.doc);
  }
  // 'rerank' (default): BM25 top-N candidates, then cosine-rerank to top-K.
  const candidates = bm25.search(claimText, BM25_POOL);
  if (!candidates.length) return [];
  const reranked = candidates.map(c => {
    const idx = bm25.docs.indexOf(c.doc);
    return { doc: c.doc, sim: cosineSimilarity(claimEmbedding, sentenceEmbeddings[idx]) };
  });
  reranked.sort((a, b) => b.sim - a.sim);
  return reranked.slice(0, EVIDENCE_TOP_K).map(r => r.doc);
}

async function verifyClaims(claims, retrievalMethod) {
  for (let i = 0; i < claims.length; i++) {
    const claim = claims[i];
    const claimEmbedding = retrievalMethod !== 'bm25' ? await embed(claim.text) : null;
    const evidenceDocs = await retrieve(claim.text, retrievalMethod, claimEmbedding);

    if (!evidenceDocs.length) {
      self.postMessage({
        type: 'CLAIM_RESULT', claimIndex: claim.index,
        result: { status: 'UNSUPPORTED', confidence: 'LOW', evidence: 'not found in sources', source: 'none', cosineSim: 0, nliScores: null },
      });
      continue;
    }

    let topCosine = 0;
    if (retrievalMethod !== 'bm25') {
      for (const d of evidenceDocs) {
        const idx = bm25.docs.indexOf(d);
        topCosine = Math.max(topCosine, cosineSimilarity(claimEmbedding, sentenceEmbeddings[idx]));
      }
    }

    // Skip the NLI step outright when retrieval couldn't find anything
    // plausibly related — matches the plan's low-cosine short-circuit,
    // scoped to only apply when embeddings actually ran.
    if (retrievalMethod !== 'bm25' && topCosine < 0.25) {
      self.postMessage({
        type: 'CLAIM_RESULT', claimIndex: claim.index,
        result: { status: 'UNSUPPORTED', confidence: 'LOW', evidence: 'not found in sources', source: 'none', cosineSim: topCosine, nliScores: null },
      });
      continue;
    }

    let best = { supported: 0, contradicted: 0, unrelated: 1 };
    let bestDoc = evidenceDocs[0];
    let bestSalience = 0;
    const scoresPerDoc = await classifyPairsBatch(classifier, evidenceDocs.map(d => d.plainText), claim.text);
    for (let d = 0; d < evidenceDocs.length; d++) {
      const doc = evidenceDocs[d];
      const scores = scoresPerDoc[d];
      const salience = Math.max(scores.supported, scores.contradicted);
      if (salience > bestSalience) {
        best = scores; bestDoc = doc; bestSalience = salience;
      }
    }

    let { status, confidence } = verdictFromScores(best, topCosine);
    status = applyNumericGuard(status, claim.text, bestDoc.plainText);
    status = applyDominantSupportGuard(status, best);
    status = applyNegationGuard(status, claim.text, best);
    self.postMessage({
      type: 'CLAIM_RESULT',
      claimIndex: claim.index,
      result: {
        status, confidence,
        evidence: status === 'UNSUPPORTED' ? 'not found in sources' : bestDoc.plainText,
        source: status === 'UNSUPPORTED' ? 'none' : bestDoc.sourceFile,
        cosineSim: topCosine,
        nliScores: best,
      },
    });
  }
  self.postMessage({ type: 'VERIFY_DONE' });
}

self.onmessage = async e => {
  const msg = e.data;
  try {
    if (msg.type === 'LOAD_MODELS') {
      await ensureModels(msg);
      self.postMessage({ type: 'MODELS_READY' });
    } else if (msg.type === 'INDEX') {
      await buildIndex(msg.sentences, msg.retrievalMethod);
      self.postMessage({ type: 'INDEXED', count: msg.sentences.length });
    } else if (msg.type === 'VERIFY') {
      await verifyClaims(msg.claims, msg.retrievalMethod);
    }
  } catch (err) {
    self.postMessage({ type: 'ERROR', message: err.message, requestType: msg.type });
  }
};
