import { splitReferenceEntries, findAllCitations } from './textProcessing.js';

// Pulls a lightweight "surname-year" fingerprint out of any citation-shaped
// text — an in-text citation like "(Rokhman et al., 2024)" or the opening of
// a reference-list entry like "Rokhman, A. (2024). Title...". Not real
// bibliographic parsing (locale-specific formats, corporate authors, and
// multi-word surnames are handled loosely at best), but good enough for a
// heuristic cross-check against an uploaded corpus with no structured
// metadata of its own. Both callers below share this so a reference-list
// entry and its in-text citation resolve to the same key.
export function extractCitationKey(text) {
  const yearMatch = text.match(/\b(\d{4})[a-z]?\b/);
  if (!yearMatch) return null;
  const before = text.slice(0, yearMatch.index);
  const nameMatch = before.match(/[A-Z][a-zA-Z'-]{2,}/);
  if (!nameMatch) return null;
  return `${nameMatch[0].toLowerCase()}-${yearMatch[1]}`;
}

// Does any uploaded source look like the paper this key cites? There's no
// structured metadata to match on, so this checks whether the surname and
// year both show up in the doc's filename or the first ~3000 characters of
// its extracted text (title page / header, where a paper's own author and
// year almost always appear) — a best-effort match, not a guarantee.
const MATCH_TEXT_WINDOW = 3000;

export function findMatchingSource(key, refs) {
  if (!key) return null;
  const [surname, year] = key.split('-');
  return refs.find(r => {
    const haystack = `${r.name} ${(r.text || r.content || '').slice(0, MATCH_TEXT_WINDOW)}`.toLowerCase();
    return haystack.includes(surname) && haystack.includes(year);
  }) || null;
}

export function extractReferenceKeys(referencesText) {
  return splitReferenceEntries(referencesText)
    .map(raw => ({ raw, key: extractCitationKey(raw) }))
    .filter(e => e.key);
}

// "X of Y cited sources found in corpus" — Y is reference-list entries that
// are actually cited somewhere in the body (not unused bibliography
// padding), X is how many of those resolve to an uploaded source.
export function computeSourceCoverage(referencesText, bodyText, refs) {
  const bodyKeys = new Set(findAllCitations(bodyText).map(extractCitationKey).filter(Boolean));
  const citedEntries = extractReferenceKeys(referencesText).filter(e => bodyKeys.has(e.key));
  const missing = citedEntries.filter(e => !findMatchingSource(e.key, refs));
  return { total: citedEntries.length, foundCount: citedEntries.length - missing.length, missing };
}

// For each UNSUPPORTED cited claim, works out *why* — genuinely absent from
// an available, readable source (full penalty, unchanged from today), cited
// from a source that was never uploaded (zero penalty — the tool simply
// can't have found it), or cited from a source that was uploaded but is
// probably not reliably text-searchable (reduced penalty). Leaves every
// other status untouched.
export function tagUnsupportedClaims(citedClaims, refs) {
  return citedClaims.map(c => {
    if (c.status !== 'UNSUPPORTED') return c;
    const key = extractCitationKey(c.citation || '');
    const source = findMatchingSource(key, refs);
    let unsupportedReason;
    if (!source) unsupportedReason = 'SOURCE_NOT_UPLOADED';
    else if (source.quality && (source.quality.largePage || source.quality.sparseText)) unsupportedReason = 'EXTRACTION_LIMITED';
    else unsupportedReason = 'NOT_FOUND_IN_SOURCE';
    return { ...c, unsupportedReason };
  });
}

// Same weighted-credit shape as the raw score (SUPPORTED=1, PARTIAL=0.5),
// but reason-aware for UNSUPPORTED: a claim whose source was never uploaded
// is excluded from the score entirely (it isn't evidence the claim is
// wrong — the tool just never got a chance to check it), one whose source
// has known extraction problems gets half credit instead of zero, and a
// genuine miss (source present and readable) keeps the full penalty.
export function computeAdjustedScore(taggedClaims) {
  let numerator = 0, denominator = 0;
  taggedClaims.forEach(c => {
    if (c.status === 'SUPPORTED') { numerator += 1; denominator += 1; }
    else if (c.status === 'PARTIAL') { numerator += 0.5; denominator += 1; }
    else if (c.status === 'CONTRADICTED') { denominator += 1; }
    else if (c.status === 'UNSUPPORTED') {
      if (c.unsupportedReason === 'SOURCE_NOT_UPLOADED') return;
      denominator += 1;
      if (c.unsupportedReason === 'EXTRACTION_LIMITED') numerator += 0.5;
    }
  });
  return denominator ? Math.round(100 * numerator / denominator) : 0;
}
