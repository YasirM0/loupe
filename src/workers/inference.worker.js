import { pipeline, env } from '@huggingface/transformers';
import { BM25Index, cosineSimilarity } from '../lib/bm25.js';
import { classifyPairsBatch, verdictFromScores, applyNumericGuard, applyNegationGuard, applyDominantSupportGuard } from '../lib/nli.js';
import { translateIdToEn } from '../lib/dictionary.id-en.js';

env.allowLocalModels = false;

let embedder = null;
let embedderName = null;
let classifier = null;
let classifierName = null;

// Indonesian support translates into English and reuses these exact
// English-tuned models rather than swapping to a multilingual model — see
// bench/run.mjs's 2026-08-30 comparison: dictionary substitution
// (dictionary.id-en.js) for retrieval and full MT (opus-mt-id-en) for NLI
// measured 100%/86% on the official test sets, identical to the English
// baseline, while a multilingual model with no translation only reached
// 33%/36%. `lang` is set once via LOAD_MODELS and read by embed/verify
// below; `translator` only loads when it's actually 'id'.
let lang = 'en';
let translator = null;
const TRANSLATOR_MODEL = 'Xenova/opus-mt-id-en';

// Cheap stopword-overlap check, not a real language-ID library — this only
// ever has to distinguish Indonesian from English within a single paper's
// own text, not general language ID. Gates the MT step specifically: an
// Indonesian paper quoting an English passage inline shouldn't have that
// passage run through an id->en model, which isn't trained to handle
// same-language passthrough and could mangle it. Dictionary substitution
// doesn't need this gate — it only ever replaces recognized Indonesian
// terms, so it's already a no-op on text that has none.
const ID_STOPWORDS = new Set(['yang', 'dan', 'tidak', 'adalah', 'dari', 'dengan', 'ini', 'itu', 'untuk', 'pada', 'dalam', 'akan', 'telah', 'oleh', 'sebagai', 'karena', 'juga', 'tersebut', 'namun', 'secara']);
const EN_STOPWORDS = new Set(['the', 'and', 'of', 'to', 'in', 'is', 'that', 'for', 'with', 'on', 'as', 'by', 'this', 'was', 'are', 'be', 'not', 'were']);
function looksIndonesian(text) {
  const words = text.toLowerCase().split(/[^a-z]+/).filter(Boolean);
  if (!words.length) return false;
  let idHits = 0, enHits = 0;
  for (const w of words) { if (ID_STOPWORDS.has(w)) idHits++; else if (EN_STOPWORDS.has(w)) enHits++; }
  return idHits > enHits;
}

function dictTranslate(text) {
  return lang === 'id' ? translateIdToEn(text) : text;
}

async function mtTranslate(text) {
  if (lang !== 'id' || !translator || !looksIndonesian(text)) return text;
  const out = await translator(text);
  return out[0].translation_text;
}

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

async function ensureModels({ embedModel, nliModel, retrievalMethod, nliDtype, embedPooling: pooling, embedQueryPrefix: queryPrefix, embedPassagePrefix: passagePrefix, lang: msgLang }) {
  lang = msgLang || 'en';
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
  if (lang === 'id' && !translator) {
    // Xenova/opus-mt-id-en's default quantized (q8) export fails session
    // creation on the current ONNX Runtime bundled with transformers.js —
    // "Missing required scale ... MatMulNBits" — a known incompatibility
    // between older QDQ-quantized exports and the v4+ runtime's optimizer
    // (transformers.js issue #1707, same failure class, different model).
    // fp32 is the one dtype every opus-mt-* export actually carries a valid
    // graph for; the ~2x larger download only happens once per browser.
    translator = await pipeline('translation', TRANSLATOR_MODEL, {
      device: 'wasm',
      dtype: 'fp32',
      progress_callback: p => self.postMessage({ type: 'MODEL_PROGRESS', model: 'translation', ...p }),
    });
  }
}

// Only ever called with a claim's text (the "query" side of retrieval) —
// see verifyClaims below, its one caller.
async function embed(text) {
  const out = await embedder(embedQueryPrefix + dictTranslate(text), { pooling: embedPooling, normalize: true });
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
  const out = await embedder(texts.map(t => embedPassagePrefix + dictTranslate(t)), { pooling: embedPooling, normalize: true });
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
    // NLI gets full-MT-translated text (not the dictionary substitution
    // embed()/embedBatch() use above) — measured separately in bench/run.mjs:
    // dictionary substitution for entailment only reached 50% (word order
    // matters for a correct verdict, not just topic overlap), while full MT
    // reached 86%, matching the English baseline exactly. The `evidence`/
    // `source` reported back below still use the original-language
    // `bestDoc.plainText` — the user should see the actual source text, not
    // a machine translation of it.
    const claimForNli = await mtTranslate(claim.text);
    const evidenceForNli = await Promise.all(evidenceDocs.map(d => mtTranslate(d.plainText)));
    const scoresPerDoc = await classifyPairsBatch(classifier, evidenceForNli, claimForNli);
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
