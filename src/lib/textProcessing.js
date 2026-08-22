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

export function extractClaims(paperText) {
  const sentences = splitSentences(paperText);
  return sentences.map((text, i) => {
    const citationMatch = text.match(CITATION_PATTERN);
    return {
      text,
      index: i,
      hasCitation: !!citationMatch,
      citationText: citationMatch ? citationMatch[0] : null,
      autoSelected: isLikelyClaim(text),
    };
  });
}
