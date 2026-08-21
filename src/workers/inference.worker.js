import { pipeline, env } from '@huggingface/transformers';
import { BM25Index, cosineSimilarity } from '../lib/bm25.js';
import { classifyPair, verdictFromScores } from '../lib/nli.js';

env.allowLocalModels = false;

let embedder = null;
let embedderName = null;
let classifier = null;
let classifierName = null;

let bm25 = null;
let sentenceEmbeddings = null; // parallel array to bm25.docs, or null if not needed

async function ensureModels({ embedModel, nliModel, retrievalMethod, nliDtype }) {
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

async function embed(text) {
  const out = await embedder(text, { pooling: 'mean', normalize: true });
  return Array.from(out.data);
}

async function buildIndex(sentences, retrievalMethod) {
  bm25 = new BM25Index(sentences);
  sentenceEmbeddings = null;
  if (retrievalMethod !== 'bm25') {
    sentenceEmbeddings = [];
    for (let i = 0; i < sentences.length; i++) {
      sentenceEmbeddings.push(await embed(sentences[i].text));
      if (i % 10 === 0 || i === sentences.length - 1) {
        self.postMessage({ type: 'INDEX_PROGRESS', done: i + 1, total: sentences.length });
      }
    }
  }
}

// Which reference sentences to compare a claim against, per retrieval method.
// Top-5 (not top-3) and a wider BM25 candidate pool — more evidence for the
// NLI step to weigh per claim, trading some speed for closer-to-LLM recall.
const EVIDENCE_TOP_K = 5;
const BM25_POOL = 30;

async function retrieve(claimText, retrievalMethod) {
  if (retrievalMethod === 'bm25') {
    return bm25.search(claimText, EVIDENCE_TOP_K).map(r => r.doc);
  }
  if (retrievalMethod === 'embeddings') {
    const qEmb = await embed(claimText);
    const scored = bm25.docs.map((doc, i) => ({ doc, sim: cosineSimilarity(qEmb, sentenceEmbeddings[i]) }));
    scored.sort((a, b) => b.sim - a.sim);
    return scored.slice(0, EVIDENCE_TOP_K).map(s => s.doc);
  }
  // 'rerank' (default): BM25 top-N candidates, then cosine-rerank to top-K.
  const candidates = bm25.search(claimText, BM25_POOL);
  if (!candidates.length) return [];
  const qEmb = await embed(claimText);
  const reranked = candidates.map(c => {
    const idx = bm25.docs.indexOf(c.doc);
    return { doc: c.doc, sim: cosineSimilarity(qEmb, sentenceEmbeddings[idx]) };
  });
  reranked.sort((a, b) => b.sim - a.sim);
  return reranked.slice(0, EVIDENCE_TOP_K).map(r => r.doc);
}


async function verifyClaims(claims, retrievalMethod) {
  for (let i = 0; i < claims.length; i++) {
    const claim = claims[i];
    const evidenceDocs = await retrieve(claim.text, retrievalMethod);

    if (!evidenceDocs.length) {
      self.postMessage({
        type: 'CLAIM_RESULT', claimIndex: claim.index,
        result: { status: 'UNSUPPORTED', confidence: 'LOW', evidence: 'not found in sources', source: 'none', cosineSim: 0, nliScores: null },
      });
      continue;
    }

    let topCosine = 0;
    if (retrievalMethod !== 'bm25') {
      const qEmb = await embed(claim.text);
      for (const d of evidenceDocs) {
        const idx = bm25.docs.indexOf(d);
        topCosine = Math.max(topCosine, cosineSimilarity(qEmb, sentenceEmbeddings[idx]));
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
    for (const doc of evidenceDocs) {
      const scores = await classifyPair(classifier, doc.plainText, claim.text);
      const salience = Math.max(scores.supported, scores.contradicted);
      if (salience > bestSalience) {
        best = scores; bestDoc = doc; bestSalience = salience;
      }
    }

    const { status, confidence } = verdictFromScores(best, topCosine);
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
