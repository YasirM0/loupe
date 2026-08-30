// Larger adversarial stress test for English and Indonesian, run through the
// actual production code path end to end — chunking (extractClaims/
// chunkReferenceIntoSentences from src/lib/textProcessing.js), retrieval
// (BM25Index + cosine rerank from src/lib/bm25.js, the 'rerank' default:
// BM25 pool 30 -> cosine rerank -> top 5), NLI (classifyPairsBatch/
// verdictFromScores/applyNumericGuard/applyDominantSupportGuard/
// applyNegationGuard from src/lib/nli.js), and the same translation split
// inference.worker.js uses for Indonesian (dictTranslate for embeddings,
// full MT via opus-mt-id-en for NLI). This is a full-paper simulation, not
// isolated claim/evidence pairs like NLI_TESTSET/RETRIEVAL_TESTSET in
// bench/run.mjs — it exercises retrieval competition across a whole
// multi-document corpus the way a real upload does.
//
//   node bench/stress-run.mjs
import { pipeline, env } from '@huggingface/transformers';
import { BM25Index, cosineSimilarity } from '../src/lib/bm25.js';
import { classifyPairsBatch, verdictFromScores, applyNumericGuard, applyNegationGuard, applyDominantSupportGuard, applyDominantContradictionGuard } from '../src/lib/nli.js';
import { chunkReferenceIntoSentences } from '../src/lib/textProcessing.js';
import { translateIdToEn } from '../src/lib/dictionary.id-en.js';
import { EN_SOURCES, EN_CLAIMS, ID_SOURCES, ID_CLAIMS } from './stress-testset.mjs';

env.allowLocalModels = false;

const EMBED_MODEL = 'Xenova/all-MiniLM-L12-v2';
const NLI_MODEL = 'Xenova/deberta-v3-base-tasksource-nli';
const EVIDENCE_TOP_K = 5;
const BM25_POOL = 30;

async function runStress({ label, sources, claims, translateForEmbed = t => t, translateForNli = async t => t }) {
  console.log(`\n=== STRESS TEST: ${label} ===`);

  const embedder = await pipeline('feature-extraction', EMBED_MODEL, { device: 'cpu' });
  const classifier = await pipeline('zero-shot-classification', NLI_MODEL, { device: 'cpu' });
  const embed = async text => Array.from((await embedder(translateForEmbed(text), { pooling: 'mean', normalize: true })).data);

  const sentences = sources.flatMap(s => chunkReferenceIntoSentences(s.name, s.text));
  const bm25 = new BM25Index(sentences);
  const sentenceEmbeddings = [];
  for (const s of sentences) sentenceEmbeddings.push(await embed(s.text));

  let correct = 0;
  const details = [];

  for (const item of claims) {
    const claimEmbedding = await embed(item.claim);
    const candidates = bm25.search(item.claim, BM25_POOL);
    let evidenceDocs;
    if (!candidates.length) {
      evidenceDocs = [];
    } else {
      const reranked = candidates.map(c => {
        const idx = bm25.docs.indexOf(c.doc);
        return { doc: c.doc, sim: cosineSimilarity(claimEmbedding, sentenceEmbeddings[idx]) };
      });
      reranked.sort((a, b) => b.sim - a.sim);
      evidenceDocs = reranked.slice(0, EVIDENCE_TOP_K).map(r => r.doc);
    }

    let status, topCosine = 0, bestDoc = null, best = null;
    if (!evidenceDocs.length) {
      status = 'UNSUPPORTED';
    } else {
      for (const d of evidenceDocs) {
        const idx = bm25.docs.indexOf(d);
        topCosine = Math.max(topCosine, cosineSimilarity(claimEmbedding, sentenceEmbeddings[idx]));
      }
      if (topCosine < 0.25) {
        status = 'UNSUPPORTED';
      } else {
        const claimForNli = await translateForNli(item.claim);
        const evidenceForNli = await Promise.all(evidenceDocs.map(d => translateForNli(d.plainText)));
        const scoresPerDoc = await classifyPairsBatch(classifier, evidenceForNli, claimForNli);
        best = { supported: 0, contradicted: 0, unrelated: 1 };
        bestDoc = evidenceDocs[0];
        let bestSalience = 0;
        for (let d = 0; d < evidenceDocs.length; d++) {
          const scores = scoresPerDoc[d];
          const salience = Math.max(scores.supported, scores.contradicted);
          if (salience > bestSalience) { best = scores; bestDoc = evidenceDocs[d]; bestSalience = salience; }
        }
        ({ status } = verdictFromScores(best, topCosine));
        status = applyNumericGuard(status, item.claim, bestDoc.plainText);
        status = applyDominantSupportGuard(status, best);
        status = applyDominantContradictionGuard(status, best);
        status = applyNegationGuard(status, item.claim, best);
      }
    }

    const isSpecial = item.expected === 'SPECIAL';
    const ok = isSpecial ? null : status === item.expected;
    if (ok) correct++;
    details.push({ claim: item.claim, expected: item.expected, got: status, trap: item.trap, source: bestDoc?.sourceFile, special: isSpecial, topCosine, best });
    const tag = isSpecial ? 'DIAG' : (ok ? 'OK  ' : 'MISS');
    const scoreStr = best ? `sup=${best.supported.toFixed(2)} con=${best.contradicted.toFixed(2)} unr=${best.unrelated.toFixed(2)}` : 'no-nli';
    console.log(`  ${tag} expected=${item.expected} got=${status}  cos=${topCosine.toFixed(3)} ${scoreStr}  [${item.trap}]  "${item.claim.slice(0, 70)}"`);
    if (isSpecial) console.log(`       -> matched source: ${bestDoc?.sourceFile || 'none'}  cosine=${topCosine.toFixed(3)}`);
  }

  const scored = claims.filter(c => c.expected !== 'SPECIAL').length;
  const pct = Math.round((correct / scored) * 100);
  console.log(`  -> ${correct}/${scored} = ${pct}%  (${claims.length - scored} diagnostic case(s) excluded from scoring)`);
  return { pct, correct, scored, details };
}

const idMtTranslator = await pipeline('translation', 'Xenova/opus-mt-id-en', { device: 'cpu' });
const mtTranslate = async text => (await idMtTranslator(text))[0].translation_text;

const enResult = await runStress({ label: 'English', sources: EN_SOURCES, claims: EN_CLAIMS });
const idResult = await runStress({
  label: 'Indonesian', sources: ID_SOURCES, claims: ID_CLAIMS,
  translateForEmbed: translateIdToEn, translateForNli: mtTranslate,
});

console.log('\n\n=== SUMMARY ===');
console.log(`English:    ${enResult.correct}/${enResult.scored} = ${enResult.pct}%`);
console.log(`Indonesian: ${idResult.correct}/${idResult.scored} = ${idResult.pct}%`);

const enSpecial = enResult.details.find(d => d.special);
const idSpecial = idResult.details.find(d => d.special);
console.log('\nDiagnostic case (wrong-source citation, claim 9):');
console.log(`  English:    got ${enSpecial.got}, matched "${enSpecial.source}"`);
console.log(`  Indonesian: got ${idSpecial.got}, matched "${idSpecial.source}"`);

console.log('\nMisses by trap category:');
for (const r of [{ label: 'English', ...enResult }, { label: 'Indonesian', ...idResult }]) {
  const misses = r.details.filter(d => !d.special && d.got !== d.expected);
  if (!misses.length) { console.log(`  ${r.label}: none`); continue; }
  misses.forEach(m => console.log(`  ${r.label}: [${m.trap}] expected=${m.expected} got=${m.got} "${m.claim.slice(0, 60)}"`));
}
