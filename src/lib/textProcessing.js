// Abbreviations that end in a period but don't end a sentence — without this
// list, "et al. found" or "Fig. 3 shows" would get split mid-thought.
const ABBREVIATIONS = [
  'et al', 'e.g', 'i.e', 'vs', 'Dr', 'Mr', 'Mrs', 'Ms', 'Prof', 'Fig', 'No', 'al', 'etc',
  'Inc', 'Ltd', 'Jr', 'Sr', 'vol', 'pp', 'ed', 'eds', 'Ch', 'Sec', 'Eq',
];
const ABBR_PATTERN = new RegExp(`\\b(${ABBREVIATIONS.join('|')})\\.$`, 'i');

const MIN_SENTENCE_WORDS = 8;

// Splits on ./?/! followed by whitespace + a capital letter, then filters out
// abbreviation false-splits and fragments too short to carry real signal.
// Each sentence keeps the previous one as light overlap context, since a
// claim ("this reduction") can depend on the sentence just before it.
export function splitSentences(text) {
  const raw = text
    .replace(/\r\n/g, '\n')
    .split(/(?<=[.?!])\s+(?=[A-Z(])/);

  const sentences = [];
  let buffer = '';
  for (const piece of raw) {
    buffer = buffer ? `${buffer} ${piece}` : piece;
    const trimmedEnd = buffer.trim().split(/\s+/).slice(-1)[0] || '';
    if (ABBR_PATTERN.test(trimmedEnd)) continue; // keep accumulating, false split
    const clean = buffer.trim();
    if (clean) sentences.push(clean);
    buffer = '';
  }
  if (buffer.trim()) sentences.push(buffer.trim());

  return sentences
    .map(s => s.replace(/\s+/g, ' ').trim())
    .filter(s => s.split(/\s+/).length >= MIN_SENTENCE_WORDS);
}

// Matches a line that IS a references heading, nothing else on it (allows
// trailing punctuation/whitespace/markdown bold markers a paper's export
// step might leave behind, e.g. "**REFERENCES**").
const REFERENCES_HEADING = /^\**\s*(references|bibliography|works\s+cited)\s*\**\s*:?\s*$/i;

// Everything from a References/Bibliography/Works Cited heading to the end
// of the document is reference-list text, not body prose — scanning it for
// citation verification or uncited-claim flagging produces nonsense (a
// reference entry like "Otoritas: Jurnal Ilmu Pemerintahan, 14(2), 481-498"
// reads as an uncited numeric claim to the local rule). Only the first such
// heading counts as the boundary; a paper's own body text mentioning the
// word "references" some other way won't match since this requires the
// heading to be the *entire* line.
export function splitReferencesSection(text) {
  const lines = text.split('\n');
  let splitAt = -1;
  for (let i = 0; i < lines.length; i++) {
    if (REFERENCES_HEADING.test(lines[i])) { splitAt = i; break; }
  }
  if (splitAt === -1) return { body: text, referencesText: '', found: false, entryCount: 0 };
  const body = lines.slice(0, splitAt).join('\n');
  const referencesText = lines.slice(splitAt + 1).join('\n').trim();
  return { body, referencesText, found: true, entryCount: countReferenceEntries(referencesText) };
}

// Reference lists are usually either blank-line-separated entries (common in
// exported/formatted text) or one entry per line with no blank separators —
// falls back to counting lines that look like they start a citation entry
// (an author-list opening, or a "[N]" numeric marker) rather than just
// counting every non-blank line, which would overcount wrapped entries.
export function countReferenceEntries(referencesText) {
  return splitReferenceEntries(referencesText).length;
}

export function splitReferenceEntries(referencesText) {
  if (!referencesText.trim()) return [];
  const paras = referencesText.split(/\n\s*\n/).map(p => p.replace(/\s+/g, ' ').trim()).filter(Boolean);
  if (paras.length > 1) return paras;
  const lines = referencesText.split('\n').map(l => l.trim()).filter(Boolean);
  const entryStarts = lines.filter(l => /^[A-Z][a-zA-Z'-]+,/.test(l) || /^\[\d+\]/.test(l));
  return entryStarts.length ? entryStarts : lines;
}

// One doc = one source file's sentences, each carrying an overlap prefix
// (the previous sentence) so retrieval doesn't lose cross-sentence context.
export function chunkReferenceIntoSentences(sourceFile, text) {
  const sentences = splitSentences(text);
  return sentences.map((s, i) => ({
    text: i > 0 ? `${sentences[i - 1]} ${s}` : s,
    plainText: s,
    sentenceIndex: i,
    sourceFile,
  }));
}

// Author-date (APA-ish and Chicago author-date) plus bracketed numeric
// citations (Vancouver/IEEE-style, also how a loose Chicago notes style
// often reads once footnote markers get flattened to plain text).
//
// AUTHOR allows up to two lowercase surname particles ("de", "van der", "la",
// etc.) before the capitalized surname — a plain `[A-Z]...` start rejects
// real, common names like "de Haas" outright, which a synthetic-paper stress
// test caught missing one of this app's own bundled reference authors.
const PARTICLE = `(?:de|van|von|der|den|la|le|di|da|dos|das|del|al)`;
const AUTHOR = `(?:${PARTICLE}\\s+){0,2}[A-Z][a-zA-Z'-]*`;
// "et al." previously required a name to follow it (modeled on "& Name"),
// when in real usage it replaces the remaining names rather than introducing
// another one — "(Rokhman et al., 2024)" never matched. It stands alone, and
// a comma-separated author list ("Smith, Jones, & Brown, 2020") is
// supported too, not just a single "&"/"and" pair.
const AUTHOR_LIST = `${AUTHOR}(?:,\\s*${AUTHOR})*(?:,?\\s+(?:&|and)\\s+${AUTHOR})?(?:\\s+et\\s+al\\.?)?`;
const YEAR = `\\d{4}[a-z]?`;
// Trailing locators ("p. 45", "pp. 12-15") or a secondary/nested attribution
// ("as cited in Smith, 2020") that would otherwise sit between the year and
// the closing paren and silently break the match.
const LOCATOR = `(?:,\\s*p{1,2}\\.?\\s*\\d+(?:[-–]\\d+)?|,\\s*as\\s+cited\\s+in\\s+${AUTHOR_LIST},?\\s+${YEAR})?`;
// One or more semicolon-joined citations in a single parenthetical
// ("(Arango, 2000; de Haas, 2010; Massey, 2019)") — the common way to stack
// multiple sources, previously unmatched as a whole even when every entry
// individually would have matched alone.
const PAREN_CITATION = `\\(${AUTHOR_LIST},?\\s+${YEAR}${LOCATOR}(?:;\\s*${AUTHOR_LIST},?\\s+${YEAR}${LOCATOR})*\\)`;
// Narrative citations ("de Haas (2021) argues...", "King et al.'s (2016)
// findings...") put the author outside the parens and only the year inside
// — by far the most common academic citation style, and previously entirely
// invisible since the old pattern required an author token inside the same
// parens as the year.
const NARRATIVE_CITATION = `${AUTHOR_LIST}(?:'s)?\\s*\\(${YEAR}${LOCATOR}\\)`;
// A narrow, signal-worded fallback for the one remaining narrative style with
// no parens or brackets at all ("According to Smith, 2020, ..."). Only
// matches right after "according to"/"per" specifically (case-insensitive on
// just that phrase — NOT the whole pattern, which would defeat AUTHOR's
// capitalization requirement), rather than any bare "Name, YYYY", to keep
// the false-positive rate low.
const SIGNAL_BARE_CITATION = `(?:[Aa]ccording\\s+to|[Pp]er)\\s+${AUTHOR_LIST},?\\s+${YEAR}`;
const CITATION_PATTERN = new RegExp(
  `${PAREN_CITATION}|${NARRATIVE_CITATION}|${SIGNAL_BARE_CITATION}|\\[\\d+(?:[-–,]\\s*\\d+)*\\]`
);
const NUMBER_PATTERN = /\b\d+(\.\d+)?%|\b\d{2,}(\.\d+)?\b/;
// shown/revealed/reported/argued/concluded added after a stress test found a
// fabricated claim using "has shown" slip through undetected — only the
// present-tense "shows?" was covered, missing the past-participle form any
// perfect-tense phrasing ("has shown", "have shown") actually uses.
// documented/identified/flagged/observed/noted added after the same test
// found a true-but-uncited empirical claim ("...incorrectly flagged
// thousands of families...") invisible on every heuristic axis — these are
// common report-verbs for exactly the kind of checkable finding a citation
// checker exists to catch, and a false positive here only costs one extra
// item in the "no citation" review list, not a false verification result.
const CAUSAL_VERBS = /\b(shows?|shown|demonstrat(?:es?|ed)|reveal(?:s|ed)?|indicat(?:es?|ed)|proves?|proved|confirms?|confirmed|found\s+that|reports?|reported|argues?|argued|concludes?|concluded|suggests?\s+that|document(?:s|ed)?|identif(?:y|ies|ied)|flags?|flagged|observ(?:es?|ed)|noted?\s+that)\b/i;
const COMPARATIVE = /\b(more than|less than|higher|lower|increased?|decreased?|significantly)\b/i;

// A sentence counts as a "claim" worth checking if it looks like it's
// asserting something specific and checkable, not just narrating or arguing.
export function isLikelyClaim(sentence) {
  return CITATION_PATTERN.test(sentence) || NUMBER_PATTERN.test(sentence)
    || CAUSAL_VERBS.test(sentence) || COMPARATIVE.test(sentence);
}

// Every in-text citation substring found in a stretch of body text — used
// by sourceCoverage.js to work out which reference-list entries are actually
// cited in the paper (vs. bibliography padding), without exposing
// CITATION_PATTERN itself outside this module.
export function findAllCitations(text) {
  return text.match(new RegExp(CITATION_PATTERN.source, CITATION_PATTERN.flags + 'g')) || [];
}

// Split into paragraphs first (same boundary chunkText uses) so each claim
// can carry its paragraph position — the uncited-claim filter below needs
// that to check whether a citation on the sentence right before/after it in
// the same paragraph should be treated as covering it too. Sentence `index`
// stays a flat, unique counter (still what inference.worker.js matches
// results back against), just assigned paragraph-by-paragraph instead of as
// one whole-document stream.
export function extractClaims(paperText) {
  const paragraphs = paperText.split(/\n\s*\n/).map(p => p.trim()).filter(Boolean);
  const claims = [];
  let index = 0;
  paragraphs.forEach((para, paragraphIndex) => {
    const sentences = splitSentences(para);
    sentences.forEach((text, indexInParagraph) => {
      const citationMatch = text.match(CITATION_PATTERN);
      claims.push({
        text,
        index: index++,
        paragraphIndex,
        indexInParagraph,
        hasCitation: !!citationMatch,
        citationText: citationMatch ? citationMatch[0] : null,
        autoSelected: isLikelyClaim(text),
      });
    });
  });
  return claims;
}

// Sentence-initial discourse/signposting markers — analytical or
// transitional sentences academic convention doesn't expect a citation on,
// even when they happen to contain a number or reasoning verb ("Section 3
// provides data on...", "These findings suggest a broader pattern").
const DISCOURSE_MARKERS = /^(this\s+(paper|article|study|analysis|section)\s+(argues?|shows?|suggests?|means?|is|reflects?)|this\s+(suggests?|means?|is|reflects?)|in\s+other\s+words|as\s+shown|section\s+\d+|these\s+findings|the\s+combined|together\s+these|the\s+result\s+is)\b/i;

function startsWithDiscourseMarker(sentence) {
  return DISCOURSE_MARKERS.test(sentence.trim());
}

// Study/finding-context nouns — when a sentence's only trigger for
// isLikelyClaim is a bare reasoning verb (no number, no comparison), this is
// what distinguishes "the study found that access improved" (a checkable,
// externally-attributed finding worth flagging) from "this suggests trust
// erodes over time" (the paper's own interpretive commentary).
const STUDY_CONTEXT = /\b(stud(?:y|ies)|research|survey|data(?:set)?|evidence|finding(?:s)?|report(?:s)?|analysis)\b/i;
const SELF_REFERENTIAL_SUBJECT = /^(this|these|that|those|such|it)\b/i;

// Whether a sentence that already tripped isLikelyClaim actually asserts
// something concrete and checkable, vs. purely interpretive/argumentative
// language riding along on the same reasoning-verb regex. Deliberately
// narrow on the causal-verb-only branch: broadening this to *any* bare
// causal-verb match would re-suppress externally-attributed findings phrased
// without a study noun (see the "incorrectly flagged thousands of families"
// regression note above CAUSAL_VERBS) — it only cuts sentences that are also
// self-referential in subject ("this/these/it..."), which the discourse
// study noun/self-reference distinction leaves everything else alone.
function hasConcreteEmpiricalSignal(sentence) {
  if (NUMBER_PATTERN.test(sentence)) return true;
  if (COMPARATIVE.test(sentence)) return true;
  if (!CAUSAL_VERBS.test(sentence)) return false;
  if (STUDY_CONTEXT.test(sentence)) return true;
  return !SELF_REFERENTIAL_SUBJECT.test(sentence.trim());
}

// Exported separately from extractUncitedClaims so the LLM-provider path
// (whose uncited claims come from the model, not this local rule) can still
// bucket its own results into the same three report groups for display.
export function categorizeUncitedClaim(sentence) {
  if (NUMBER_PATTERN.test(sentence)) return 'STATISTIC';
  if (COMPARATIVE.test(sentence)) return 'COMPARATIVE';
  return 'INTERPRETIVE';
}

// Turns the raw `isLikelyClaim`-flagged, uncited candidates from
// `extractClaims` into the actual "no citation" review list — layering on
// the checks that keep the local rule from drowning a real issue in noise:
// discourse-marker sentences, sentences within one sentence of a citation in
// the same paragraph (citation-inheritance window), and sentences with no
// concrete empirical signal once discourse framing is stripped out.
export function extractUncitedClaims(allClaims) {
  const out = [];
  allClaims.forEach((c, i) => {
    if (!c.autoSelected || c.hasCitation) return;
    if (startsWithDiscourseMarker(c.text)) return;

    const prev = allClaims[i - 1];
    const next = allClaims[i + 1];
    const prevCites = prev && prev.paragraphIndex === c.paragraphIndex && prev.hasCitation;
    const nextCites = next && next.paragraphIndex === c.paragraphIndex && next.hasCitation;
    if (prevCites || nextCites) return;

    if (!hasConcreteEmpiricalSignal(c.text)) return;

    out.push({
      claim: c.text,
      note: 'Flagged by local rules (number, comparison, or reasoning verb) — no citation attached.',
      category: categorizeUncitedClaim(c.text),
    });
  });
  return out;
}
