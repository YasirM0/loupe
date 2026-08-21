// A small hand-written test set covering the verdict types and known failure
// modes for citation-verification NLI (direct match, paraphrase, numeric
// match/mismatch, partial overlap, negation, absence, misleading similarity).
// Not a rigorous academic benchmark — just enough real signal to rank the
// model options honestly instead of guessing at percentages.

export const NLI_TESTSET = [
  // Direct match
  { evidence: "The treatment reduced symptoms by 40% in the treated group.", claim: "The treatment reduced symptoms by 40%.", expected: "SUPPORTED" },
  { evidence: "Researchers observed that mice exposed to the compound showed reduced tumor growth.", claim: "The compound reduced tumor growth in mice.", expected: "SUPPORTED" },
  // Paraphrase / synonym match
  { evidence: "Participants who received the intervention reported a marked decline in anxiety scores.", claim: "The intervention lowered anxiety in participants.", expected: "SUPPORTED" },
  { evidence: "The algorithm achieved a classification accuracy of 94.2% on the held-out test set.", claim: "The model correctly classified about 94% of test examples.", expected: "SUPPORTED" },
  // Numeric match
  { evidence: "Enrollment reached 312 participants across five sites by the end of 2022.", claim: "The study enrolled 312 participants.", expected: "SUPPORTED" },
  // Numeric mismatch (subtle, should NOT be marked supported)
  { evidence: "The drug lowered blood pressure by 12 mmHg on average.", claim: "The drug lowered blood pressure by 20 mmHg on average.", expected: "CONTRADICTED" },
  // Partial overlap
  { evidence: "The program improved reading scores among younger children, though effects for older students were unclear.", claim: "The program improved reading scores across all age groups.", expected: "PARTIAL" },
  { evidence: "Sales grew in North America but declined in every other region.", claim: "The company saw global sales growth.", expected: "PARTIAL" },
  // Negation / contradiction
  { evidence: "No significant difference in survival rate was observed between the two groups.", claim: "The treatment significantly improved survival rate.", expected: "CONTRADICTED" },
  { evidence: "The follow-up study found the original result did not replicate.", claim: "The original finding was later confirmed by follow-up research.", expected: "CONTRADICTED" },
  // Complete absence (evidence about something else entirely)
  { evidence: "The manufacturing facility relocated to a larger site in 2019 to increase production capacity.", claim: "The treatment reduced symptoms by 40%.", expected: "UNSUPPORTED" },
  { evidence: "Coffee consumption has increased steadily across urban populations over the past decade.", claim: "The algorithm achieved 94% accuracy.", expected: "UNSUPPORTED" },
  // Misleading similarity — shares vocabulary but opposite meaning
  { evidence: "The vaccine showed no measurable reduction in transmission rates among vaccinated individuals.", claim: "The vaccine reduced transmission rates.", expected: "CONTRADICTED" },
  { evidence: "Critics argue the policy increased inequality rather than reducing it, despite its stated goals.", claim: "The policy reduced inequality, as intended.", expected: "CONTRADICTED" },
];

// For the embedding/retrieval benchmark: each item is a claim plus a small
// pool of candidate sentences, exactly one of which is the true match — the
// rest are plausible-looking distractors sharing vocabulary or topic.
export const RETRIEVAL_TESTSET = [
  {
    claim: "The treatment reduced symptoms by 40% in the treated group.",
    correctIndex: 0,
    pool: [
      "In our 2021 clinical trial, patients receiving the treatment showed a 40 percent reduction in reported symptoms.",
      "The placebo group showed a 5% reduction in symptoms over the same period.",
      "The treatment was administered twice daily for eight weeks.",
      "Symptom severity was measured using a standardized 10-point clinical scale.",
    ],
  },
  {
    claim: "The model correctly classified about 94% of test examples.",
    correctIndex: 2,
    pool: [
      "The dataset contained 10,000 labeled training examples across twelve categories.",
      "Training took approximately six hours on a single GPU.",
      "On the held-out test set, the classifier achieved 94.2% accuracy.",
      "A baseline logistic regression model achieved 71% accuracy on the same task.",
    ],
  },
  {
    claim: "The company saw global sales growth.",
    correctIndex: 1,
    pool: [
      "The company's headquarters moved to a new office in downtown Seattle.",
      "Sales grew 8% in North America but declined 3% in Europe and 6% in Asia.",
      "The CEO announced a new product line at the annual shareholder meeting.",
      "Employee headcount increased by 200 over the fiscal year.",
    ],
  },
  {
    claim: "No significant difference in survival rate was found between treatment groups.",
    correctIndex: 3,
    pool: [
      "Patients in the treatment group reported fewer side effects than the control group.",
      "The trial was conducted across fourteen hospitals in three countries.",
      "Median follow-up time was 18 months for both cohorts.",
      "Survival analysis showed no statistically significant difference between the treatment and control arms (p=0.42).",
    ],
  },
  {
    claim: "The vaccine reduced transmission rates.",
    correctIndex: 0,
    pool: [
      "Transmission rates among vaccinated individuals were statistically indistinguishable from unvaccinated individuals.",
      "The vaccine was approved for emergency use in early 2021.",
      "Manufacturing capacity was scaled up to meet global demand.",
      "Side effects were generally mild and resolved within 48 hours.",
    ],
  },
  {
    claim: "The intervention lowered anxiety in participants.",
    correctIndex: 3,
    pool: [
      "The intervention consisted of eight weekly group sessions.",
      "Participants were recruited from three community health centers.",
      "Dropout rate was 12% over the course of the study.",
      "Anxiety scores on the standardized inventory dropped by an average of 6.3 points post-intervention, a marked decline from baseline.",
    ],
  },
];
