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

// Author-date parenthetical (APA-ish and Chicago author-date — Chicago just
// tends to omit the comma before the year, already optional here) plus
// bracketed numeric citations (Vancouver/IEEE-style, also how a loose
// Chicago notes style often reads once footnote markers get flattened to
// plain text). "et al." previously and incorrectly required a name to
// follow it (modeled on "& Name"/"and Name"), when in real usage "et al."
// replaces the remaining names rather than introducing another one —
// "(Rokhman et al., 2024)" never matched. It now stands alone, and a
// comma-separated author list ("Smith, Jones, & Brown, 2020") is supported
// too, not just a single "&"/"and" pair.
const AUTHOR = `[A-Z][a-zA-Z'-]*`;
const CITATION_PATTERN = new RegExp(
  `\\(${AUTHOR}(?:,\\s*${AUTHOR})*(?:,?\\s+(?:&|and)\\s+${AUTHOR})?(?:\\s+et\\s+al\\.?)?,?\\s+\\d{4}[a-z]?\\)` +
  `|\\[\\d+(?:[-–,]\\s*\\d+)*\\]`
);
const NUMBER_PATTERN = /\b\d+(\.\d+)?%|\b\d{2,}(\.\d+)?\b/;
const CAUSAL_VERBS = /\b(shows?|demonstrat(?:es?|ed)|reveals?|indicat(?:es?|ed)|proves?|confirms?|found that|suggests?\s+that)\b/i;
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
