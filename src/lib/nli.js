// Shared between the inference worker and the benchmark script, so the
// numbers shown in the UI are measured from the exact logic that actually
// runs verification — not a similar-looking copy that could drift.
export const NLI_LABELS = ['supports', 'contradicts', 'is unrelated to'];

// The claim text is embedded directly in the hypothesis template (not left
// as a generic "This example is {label}" check) so the model actually
// compares the evidence against this specific claim, not just judging the
// evidence's tone in the abstract.
export async function classifyPair(classifier, evidenceText, claimText) {
  const safeClaim = claimText.replace(/"/g, "'");
  const result = await classifier(evidenceText, NLI_LABELS, {
    hypothesis_template: `This evidence {} the claim that "${safeClaim}".`,
  });
  const scores = {};
  result.labels.forEach((label, i) => { scores[label] = result.scores[i]; });
  return {
    supported: scores['supports'] || 0,
    contradicted: scores['contradicts'] || 0,
    unrelated: scores['is unrelated to'] || 0,
  };
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
