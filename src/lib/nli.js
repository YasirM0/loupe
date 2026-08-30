// Shared between the inference worker and the benchmark script, so the
// numbers shown in the UI are measured from the exact logic that actually
// runs verification — not a similar-looking copy that could drift.
export const NLI_LABELS = ['supports', 'contradicts', 'is unrelated to'];

function extractScores(result) {
  const scores = {};
  result.labels.forEach((label, i) => { scores[label] = result.scores[i]; });
  return {
    supported: scores['supports'] || 0,
    contradicted: scores['contradicts'] || 0,
    unrelated: scores['is unrelated to'] || 0,
  };
}

// The claim text is embedded directly in the hypothesis template (not left
// as a generic "This example is {label}" check) so the model actually
// compares the evidence against this specific claim, not just judging the
// evidence's tone in the abstract.
export async function classifyPair(classifier, evidenceText, claimText) {
  const safeClaim = claimText.replace(/"/g, "'");
  const result = await classifier(evidenceText, NLI_LABELS, {
    hypothesis_template: `This evidence {} the claim that "${safeClaim}".`,
  });
  return extractScores(result);
}

// Same comparison as classifyPair, batched across multiple evidence texts
// against one claim in a single model call — verified to produce the same
// scores as calling classifyPair once per text (floating-point noise only).
export async function classifyPairsBatch(classifier, evidenceTexts, claimText) {
  const safeClaim = claimText.replace(/"/g, "'");
  const results = await classifier(evidenceTexts, NLI_LABELS, {
    hypothesis_template: `This evidence {} the claim that "${safeClaim}".`,
  });
  return results.map(extractScores);
}

export function verdictFromScores(agg, topCosine) {
  let status, confidence;
  if (agg.contradicted > 0.75) status = 'CONTRADICTED';
  else if (agg.supported > 0.75) status = 'SUPPORTED';
  else if (agg.supported >= 0.40) status = 'PARTIAL';
  else status = 'UNSUPPORTED';

  const maxScore = Math.max(agg.supported, agg.contradicted);
  if (maxScore > 0.80 && topCosine > 0.70) confidence = 'HIGH';
  else if (maxScore >= 0.60 || topCosine >= 0.40) confidence = 'MEDIUM';
  else confidence = 'LOW';

  return { status, confidence };
}

const PERCENT_PATTERN = /\b(\d+(?:\.\d+)?)%/g;

function extractPercents(text) {
  return [...text.matchAll(PERCENT_PATTERN)].map(m => parseFloat(m[1]));
}

// NLI models judge topic/vocabulary alignment far more reliably than the
// actual magnitude of a number embedded in prose — a stress test found every
// NLI model Loupe ships (and every new candidate tried) called a claim
// SUPPORTED when it cited "roughly 60%" against evidence that actually said
// "97.68%". This is a deterministic backstop for that specific failure mode:
// if the claim states a percentage that doesn't appear (within a small
// rounding tolerance) anywhere in the evidence actually driving the verdict,
// a SUPPORTED/PARTIAL call can't be trusted on the number, so it's demoted
// to CONTRADICTED. Scoped to percentages only (the clearest same-metric
// signal) and to SUPPORTED/PARTIAL only — UNSUPPORTED already correctly
// means "evidence doesn't address this," which an absent number shouldn't
// override into a false contradiction.
const PERCENT_TOLERANCE = 3;

export function applyNumericGuard(status, claimText, evidenceText) {
  if (status !== 'SUPPORTED' && status !== 'PARTIAL') return status;
  const claimPercents = extractPercents(claimText);
  if (!claimPercents.length) return status;
  const evidencePercents = extractPercents(evidenceText);
  if (!evidencePercents.length) return status;
  const anyMatch = claimPercents.some(cp => evidencePercents.some(ep => Math.abs(cp - ep) <= PERCENT_TOLERANCE));
  return anyMatch ? status : 'CONTRADICTED';
}

// A claim asserting the *complete absence* of something ("no sense of X
// whatsoever", "entirely free of Y") that the model calls UNSUPPORTED is a
// distinct failure pattern from a claim that's merely fabricated: the
// evidence usually does address the topic and contradicts the absolute
// denial, but the model's raw scores land just under the CONTRADICTED
// threshold rather than clearing it outright.
//
// A blanket "moderate-but-dominant contradiction" rule was tried and
// rejected — simulated against real captured scores, it would have flipped
// several genuinely-UNSUPPORTED fabrications (evidence that's actually
// unrelated, not contradicting) into false CONTRADICTED calls, because
// topic-mismatch noise alone can nudge the contradicted score moderately
// upward. Gating on an explicit denial phrase in the claim text keeps this
// narrow: it only fires for the specific linguistic pattern of absolute
// denial, not any claim whose contradicted score happens to edge up.
const NEGATION_CUES = /\bno\s+.{0,60}?\s+whatsoever\b|\bnot\s+any\b|\bnone\s+of\b|\bentirely\s+free\s+of\b|\bcompletely\s+free\s+of\b|\bno\s+evidence\s+of\b/i;

export function applyNegationGuard(status, claimText, agg) {
  if (status !== 'UNSUPPORTED') return status;
  if (!NEGATION_CUES.test(claimText)) return status;
  if (agg.contradicted > agg.supported && agg.contradicted >= 0.40) return 'CONTRADICTED';
  return status;
}

// Some models (tasksource-nli in particular, likely from training on a much
// wider task mixture than narrower MNLI-style models) are systematically
// under-confident on clear matches: a stress test found several genuinely-
// SUPPORTED claims scoring supported=0.57–0.74 — clearly the dominant
// signal, often 2–5x the contradicted score — but landing in PARTIAL
// because the fixed 0.75 SUPPORTED cutoff doesn't distinguish "dominant but
// not overwhelming" from genuine partial support.
//
// The one case this can't safely separate from a real PARTIAL: a claim
// whose specific numbers are accurate but whose scope or generality is
// overstated (source says X for Latvia, claim implies X for "all Global
// South regions") scores nearly identically — the model has no way to
// signal "the numbers check out but the generalization doesn't" in three
// aggregate scores. Simulated against real captured data before shipping:
// net positive on every model tested, and *zero* regressions specifically
// for tasksource-nli (its one PARTIAL case doesn't meet this threshold),
// which is why it's shipped despite that known, documented trade-off for
// other models.
export function applyDominantSupportGuard(status, agg) {
  if (status !== 'PARTIAL') return status;
  const { supported: s, contradicted: c, unrelated: u } = agg;
  if (s >= 0.55 && s > 2 * c && s > 2 * u) return 'SUPPORTED';
  return status;
}

// Mirror of applyDominantSupportGuard for the opposite failure mode: a
// contradiction score that's clearly dominant but stops just short of the
// fixed 0.75 CONTRADICTED cutoff, landing in UNSUPPORTED — "evidence doesn't
// address this" — when the evidence actually does address it and disagrees.
// A larger adversarial stress test (bench/stress-run.mjs, paragraph-level,
// full retrieval+NLI pipeline, not isolated pairs) surfaced this repeatedly:
// numeric-mismatch and explicit-negation claims scoring contradicted=0.63-
// 0.75 with supported<0.1, called UNSUPPORTED purely because 0.75 wasn't
// cleared.
//
// Same thresholds as applyDominantSupportGuard (0.55 floor, 2x the other two
// scores) rather than a looser one — simulated first against real captured
// scores from both NLI_TESTSET/ID_NLI_TESTSET (the authoritative regression
// check, ID_NLI_TESTSET run through opus-mt-id-en first, matching production)
// and the stress set before shipping. Result: zero changes at all on either
// official 14-case test set (12/14 -> 12/14 for both), and a net accuracy
// gain on the harder stress set in both languages (bench/stress-run.mjs:
// English 45%->64%, Indonesian's one fix and one collateral case cancel out
// at 55%), at the cost of one known, narrow collateral pattern also present
// on the support side of this same guard shape: a claim about a topic the
// source explicitly didn't cover can still score contradicted-dominant from
// vocabulary/topic overlap alone, flipping a correct UNSUPPORTED into a false CONTRADICTED
// ("absence on an adjacent topic" in the stress set). This is a materially
// different risk profile than the broader "any moderately-dominant
// contradiction" idea already tried and rejected elsewhere in this file —
// that one broke the single most important trap case (a fabrication wearing
// a real author's name, correctly UNSUPPORTED); this guard's one measured
// collateral case is a real-but-uncited-scope-gap, not a citation-fraud
// case slipping through, and the official test suites show no regression at
// all from it.
export function applyDominantContradictionGuard(status, agg) {
  if (status !== 'UNSUPPORTED') return status;
  const { supported: s, contradicted: c, unrelated: u } = agg;
  if (c >= 0.55 && c > 2 * s && c > 2 * u) return 'CONTRADICTED';
  return status;
}
