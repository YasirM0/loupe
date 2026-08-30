// Abbreviations that end in a period but don't end a sentence — without this
// list, "et al. found" or "Fig. 3 shows" would get split mid-thought.
const ABBREVIATIONS_EN = [
  'et al', 'e.g', 'i.e', 'vs', 'Dr', 'Mr', 'Mrs', 'Ms', 'Prof', 'Fig', 'No', 'al', 'etc',
  'Inc', 'Ltd', 'Jr', 'Sr', 'vol', 'pp', 'ed', 'eds', 'Ch', 'Sec', 'Eq',
];
// Indonesian equivalents — dkk ("dan kawan-kawan", et al.), dll/dsb (etc.),
// dst (and so on), hlm (page). Latin-script/same punctuation conventions as
// English, so it reuses the same split algorithm below, just its own
// abbreviation list.
const ABBREVIATIONS_ID = ['dkk', 'dll', 'dst', 'hlm', 'No', 'dsb', 'Prof', 'Dr'];
const buildAbbrPattern = list => new RegExp(`\\b(${list.join('|')})\\.$`, 'i');
const ABBR_PATTERN_EN = buildAbbrPattern(ABBREVIATIONS_EN);
const ABBR_PATTERN_ID = buildAbbrPattern(ABBREVIATIONS_ID);
// Arabic abbreviations are sparser/less standardized than English's — a
// modest, real set rather than an exhaustive one (د = doctor, ص = page,
// ج = volume/part). Checked without \b (see splitSentencesAr below for why).
const ABBREVIATIONS_AR = ['د', 'ص', 'ج'];

const MIN_SENTENCE_WORDS = 8;

// Splits on ./?/! followed by whitespace + a capital letter, then filters out
// abbreviation false-splits and fragments too short to carry real signal.
// Each sentence keeps the previous one as light overlap context, since a
// claim ("this reduction") can depend on the sentence just before it.
// `lang` picks the abbreviation list (English/Indonesian both Latin-script,
// same algorithm) or hands off to the Arabic-specific splitter below, which
// can't use the same capital-letter signal at all (Arabic has no case).
export function splitSentences(text, lang = 'en') {
  const normalized = text.replace(/\r\n/g, '\n');
  if (lang === 'ar') return splitSentencesAr(normalized);

  const abbrPattern = lang === 'id' ? ABBR_PATTERN_ID : ABBR_PATTERN_EN;
  const raw = normalized.split(/(?<=[.?!])\s+(?=[A-Z(])/);

  const sentences = [];
  let buffer = '';
  for (const piece of raw) {
    buffer = buffer ? `${buffer} ${piece}` : piece;
    const trimmedEnd = buffer.trim().split(/\s+/).slice(-1)[0] || '';
    if (abbrPattern.test(trimmedEnd)) continue; // keep accumulating, false split
    const clean = buffer.trim();
    if (clean) sentences.push(clean);
    buffer = '';
  }
  if (buffer.trim()) sentences.push(buffer.trim());

  return sentences
    .map(s => s.replace(/\s+/g, ' ').trim())
    .filter(s => s.split(/\s+/).length >= MIN_SENTENCE_WORDS);
}

// No case distinction in Arabic script, so the capital-letter-after-
// punctuation signal the English/Indonesian splitter relies on has nothing
// to key on — this splits on terminal punctuation alone (`.`/`؟`/`!`),
// filtered through the (small) Arabic abbreviation list and the same
// minimum-word-count threshold. Lower precision than the case-aware
// splitter by construction (a numbered list "1." or an uncaught
// abbreviation can still cause a false split) — a real trade-off, not an
// oversight; see the plan note on Arabic support for why.
function splitSentencesAr(text) {
  const raw = text.split(/(?<=[.؟!])\s+/);
  const sentences = [];
  let buffer = '';
  for (const piece of raw) {
    buffer = buffer ? `${buffer} ${piece}` : piece;
    const trimmedEnd = buffer.trim().split(/\s+/).slice(-1)[0] || '';
    const withoutDot = trimmedEnd.endsWith('.') ? trimmedEnd.slice(0, -1) : null;
    if (withoutDot && ABBREVIATIONS_AR.includes(withoutDot)) continue; // false split, keep accumulating
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
// step might leave behind, e.g. "**REFERENCES**"). Indonesian equivalents
// ("Referensi", "Daftar Pustaka") are included unconditionally, same
// reasoning as the citation-pattern widening above — these words don't
// collide with English text, so there's no need to gate this by language.
const REFERENCES_HEADING = /^\**\s*(references|bibliography|works\s+cited|referensi|daftar\s+pustaka)\s*\**\s*:?\s*$/i;

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
export function chunkReferenceIntoSentences(sourceFile, text, lang = 'en') {
  const sentences = splitSentences(text, lang);
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
// supported too, not just a single "&"/"and" pair. "dkk." (Indonesian
// academic writing's own "et al." — "dan kawan-kawan") is accepted
// alongside it unconditionally: Latin-script Indonesian citations use the
// same author-year shape as English ones, and "dkk." doesn't realistically
// collide with English text, so this is a strict widening, not a fork.
const AUTHOR_LIST = `${AUTHOR}(?:,\\s*${AUTHOR})*(?:,?\\s+(?:&|and)\\s+${AUTHOR})?(?:\\s+(?:et\\s+al\\.?|dkk\\.?))?`;
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
// the false-positive rate low. "Menurut"/"Berdasarkan" (Indonesian for
// "according to"/"based on") widen this the same unconditional way as
// "dkk." above.
const SIGNAL_BARE_CITATION = `(?:[Aa]ccording\\s+to|[Pp]er|[Mm]enurut|[Bb]erdasarkan)\\s+${AUTHOR_LIST},?\\s+${YEAR}`;

// Arabic-script citation support, added alongside (not instead of) the
// Latin-script patterns above — Arabic papers indexed internationally very
// commonly cite with Romanized author names even in Arabic body text, so
// the existing patterns already catch a real fraction of these unmodified.
// This is a best-effort addition for the Arabic-script case specifically:
// there's no capitalization signal to distinguish "an author name" from
// "any three-letter-or-longer word" the way AUTHOR does for Latin script,
// so precision here is inherently lower (see the plan note on Arabic
// support) — some false-positive risk on phrases that happen to be
// "word(s) + (year)" without being a real citation, accepted as a
// documented trade-off rather than an oversight.
const AUTHOR_WORD_AR = `[\\u0600-\\u06FF]{3,}`;
const AUTHOR_LIST_AR = `${AUTHOR_WORD_AR}(?:\\s+${AUTHOR_WORD_AR}){0,2}(?:\\s+وآخرون)?`;
const PAREN_CITATION_AR = `\\(${AUTHOR_LIST_AR}[,،]?\\s*${YEAR}\\)`;
const NARRATIVE_CITATION_AR = `${AUTHOR_LIST_AR}\\s*\\(${YEAR}\\)`;
const SIGNAL_BARE_CITATION_AR = `(?:وفقًا\\s+لـ?|حسب)\\s+${AUTHOR_LIST_AR}[,،]?\\s*${YEAR}`;

const CITATION_PATTERN = new RegExp(
  `${PAREN_CITATION}|${NARRATIVE_CITATION}|${SIGNAL_BARE_CITATION}|\\[\\d+(?:[-–,]\\s*\\d+)*\\]`
  + `|${PAREN_CITATION_AR}|${NARRATIVE_CITATION_AR}|${SIGNAL_BARE_CITATION_AR}`
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

// Indonesian equivalents of CAUSAL_VERBS/COMPARATIVE above — Latin script,
// so the same \b(...)\b shape applies unchanged.
const CAUSAL_VERBS_ID = /\b(menunjukkan|ditunjukkan|menemukan|ditemukan(?:\s+bahwa)?|melaporkan|dilaporkan|berpendapat|berargumen|menyimpulkan|disimpulkan|menyarankan\s+bahwa|mendokumentasikan|didokumentasikan|mengidentifikasi|teridentifikasi|mengamati|teramati|mencatat(?:\s+bahwa)?|tercatat|membuktikan|terbukti|mengonfirmasi|dikonfirmasi|mengungkapkan|terungkap|menegaskan|ditegaskan)\b/i;
const COMPARATIVE_ID = /\b(lebih\s+dari|kurang\s+dari|lebih\s+tinggi|lebih\s+rendah|meningkat|menurun|secara\s+signifikan|dibandingkan\s+dengan)\b/i;

// Arabic equivalents — JavaScript's \b is defined against ASCII \w even
// with the `u` flag, so it doesn't recognize Arabic script as word
// characters at all (a boundary requires a \w/\W transition, and Arabic
// letters are always \W to the regex engine) — `\b` next to Arabic text
// silently fails to match, not just matches imprecisely. These use
// Unicode-range lookarounds instead: "not immediately preceded/followed by
// another Arabic letter" is the actual word-boundary check here.
const ARABIC_NOT_BEFORE = '(?<![\\u0600-\\u06FF])';
const ARABIC_NOT_AFTER = '(?![\\u0600-\\u06FF])';
const arWordBoundary = core => new RegExp(`${ARABIC_NOT_BEFORE}(?:${core})${ARABIC_NOT_AFTER}`);
const CAUSAL_VERBS_AR = arWordBoundary('يُظهر|تُظهر|أظهر|أظهرت|وجد|وجدت|يجد|أفاد|أفادت|يفيد|يجادل|تجادل|خلص|خلصت|يستنتج|يشير|تشير|وثّق|وثقت|يحدد|تحدد|لاحظ|لاحظت|يلاحظ|أكد|أكدت|يؤكد|كشف|كشفت|يكشف|ذكر|ذكرت|أثبت|أثبتت');
const COMPARATIVE_AR = arWordBoundary('أكثر\\s+من|أقل\\s+من|أعلى|أدنى|ارتفع|ارتفعت|انخفض|انخفضت|بشكل\\s+ملحوظ|بشكل\\s+كبير');

// A sentence counts as a "claim" worth checking if it looks like it's
// asserting something specific and checkable, not just narrating or arguing.
// `lang` picks the lexicon; CITATION_PATTERN/NUMBER_PATTERN are shared
// across languages (citation shapes and digits don't need translating).
export function isLikelyClaim(sentence, lang = 'en') {
  const lex = LEXICON[lang] || LEXICON.en;
  return CITATION_PATTERN.test(sentence) || NUMBER_PATTERN.test(sentence)
    || lex.causalVerbs.test(sentence) || lex.comparative.test(sentence);
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
export function extractClaims(paperText, lang = 'en') {
  const paragraphs = paperText.split(/\n\s*\n/).map(p => p.trim()).filter(Boolean);
  const claims = [];
  let index = 0;
  paragraphs.forEach((para, paragraphIndex) => {
    const sentences = splitSentences(para, lang);
    sentences.forEach((text, indexInParagraph) => {
      const citationMatch = text.match(CITATION_PATTERN);
      claims.push({
        text,
        index: index++,
        paragraphIndex,
        indexInParagraph,
        hasCitation: !!citationMatch,
        citationText: citationMatch ? citationMatch[0] : null,
        autoSelected: isLikelyClaim(text, lang),
      });
    });
  });
  return claims;
}

// Sentence-initial discourse/signposting markers — analytical or
// transitional sentences academic convention doesn't expect a citation on,
// even when they happen to contain a number or reasoning verb ("Section 3
// provides data on...", "These findings suggest a broader pattern").
// "together,?\s+these" (not just "together these") — caught live in the
// Indonesian equivalent below via an end-to-end run: "Bersama-sama, hasil
// ini..." (a comma-separated opener, the natural way to write it) didn't
// match a marker that only accounted for the no-comma phrasing. Fixed here
// too since the same gap exists for the English source of that sentence.
const DISCOURSE_MARKERS = /^(this\s+(paper|article|study|analysis|section)\s+(argues?|shows?|suggests?|means?|is|reflects?)|this\s+(suggests?|means?|is|reflects?)|in\s+other\s+words|as\s+shown|section\s+\d+|these\s+findings|the\s+combined|together,?\s+these|the\s+result\s+is)\b/i;

// Indonesian equivalents, same openers, Latin script so `^(...)\b` applies
// unchanged. "bersama-sama" (bare, comma-tolerant) added alongside "secara
// bersama-sama" — see the DISCOURSE_MARKERS comment above for why.
const DISCOURSE_MARKERS_ID = /^(makalah\s+ini\s+(berpendapat|berargumen|menunjukkan|menyarankan)|artikel\s+ini\s+(berpendapat|berargumen|menunjukkan)|penelitian\s+ini\s+(berpendapat|menunjukkan|menyarankan)|hal\s+ini\s+(menunjukkan|berarti)|ini\s+(menunjukkan|berarti)|dengan\s+kata\s+lain|seperti\s+yang\s+ditunjukkan|bagian\s+\d+|bab\s+\d+|temuan\s+ini|data\s+gabungan|secara\s+bersama-sama|bersama-sama,?|hasilnya\s+adalah)\b/i;

// Arabic equivalents — `^` is a position anchor, unaffected by the \b/\w
// Arabic issue, so only the trailing edge of each phrase needs the
// Unicode-lookahead boundary (ARABIC_NOT_AFTER) instead of `\b`.
const DISCOURSE_MARKERS_AR = new RegExp(
  '^(?:تجادل\\s+هذه\\s+الورقة|تناقش\\s+هذه\\s+الدراسة|يشير\\s+هذا\\s+إلى|تشير\\s+هذه\\s+الدراسة\\s+إلى|بعبارة\\s+أخرى|كما\\s+هو\\s+موضح|القسم\\s+\\d+|هذه\\s+النتائج|البيانات\\s+مجتمعة|معًا،?\\s+تشير\\s+هذه\\s+النتائج|النتيجة\\s+هي)'
  + ARABIC_NOT_AFTER
);

function startsWithDiscourseMarker(sentence, lang = 'en') {
  const lex = LEXICON[lang] || LEXICON.en;
  return lex.discourseMarkers.test(sentence.trim());
}

// Study/finding-context nouns — when a sentence's only trigger for
// isLikelyClaim is a bare reasoning verb (no number, no comparison), this is
// what distinguishes "the study found that access improved" (a checkable,
// externally-attributed finding worth flagging) from "this suggests trust
// erodes over time" (the paper's own interpretive commentary).
const STUDY_CONTEXT = /\b(stud(?:y|ies)|research|survey|data(?:set)?|evidence|finding(?:s)?|report(?:s)?|analysis)\b/i;
const SELF_REFERENTIAL_SUBJECT = /^(this|these|that|those|such|it)\b/i;

const STUDY_CONTEXT_ID = /\b(studi|penelitian|riset|survei|data|bukti|temuan|laporan|analisis)\b/i;
const SELF_REFERENTIAL_SUBJECT_ID = /^(ini|hal\s+ini|itu)\b/i;

const STUDY_CONTEXT_AR = arWordBoundary('دراسة|دراسات|بحث|أبحاث|استطلاع|بيانات|أدلة|نتائج|تقرير|تقارير|تحليل');
// `\b` right after an Arabic word doesn't fire either (no \w/\W transition
// when both the Arabic letter and a following space are \W) — a lookahead
// for whitespace-or-end stands in for it here, same reasoning as the
// trailing-edge fix in DISCOURSE_MARKERS_AR.
const SELF_REFERENTIAL_SUBJECT_AR = /^(?:هذا|هذه|ذلك|تلك)(?=\s|$)/;

// Per-language lexicon bundle. `en` is the original, unmodified behavior;
// `id`/`ar` are additive — see the plan note on Indonesian/Arabic support
// for why each pattern takes the shape it does.
const LEXICON = {
  en: { causalVerbs: CAUSAL_VERBS, comparative: COMPARATIVE, discourseMarkers: DISCOURSE_MARKERS, studyContext: STUDY_CONTEXT, selfReferential: SELF_REFERENTIAL_SUBJECT },
  id: { causalVerbs: CAUSAL_VERBS_ID, comparative: COMPARATIVE_ID, discourseMarkers: DISCOURSE_MARKERS_ID, studyContext: STUDY_CONTEXT_ID, selfReferential: SELF_REFERENTIAL_SUBJECT_ID },
  ar: { causalVerbs: CAUSAL_VERBS_AR, comparative: COMPARATIVE_AR, discourseMarkers: DISCOURSE_MARKERS_AR, studyContext: STUDY_CONTEXT_AR, selfReferential: SELF_REFERENTIAL_SUBJECT_AR },
};

// Whether a sentence that already tripped isLikelyClaim actually asserts
// something concrete and checkable, vs. purely interpretive/argumentative
// language riding along on the same reasoning-verb regex. Deliberately
// narrow on the causal-verb-only branch: broadening this to *any* bare
// causal-verb match would re-suppress externally-attributed findings phrased
// without a study noun (see the "incorrectly flagged thousands of families"
// regression note above CAUSAL_VERBS) — it only cuts sentences that are also
// self-referential in subject ("this/these/it..."), which the discourse
// study noun/self-reference distinction leaves everything else alone.
function hasConcreteEmpiricalSignal(sentence, lang = 'en') {
  const lex = LEXICON[lang] || LEXICON.en;
  if (NUMBER_PATTERN.test(sentence)) return true;
  if (lex.comparative.test(sentence)) return true;
  if (!lex.causalVerbs.test(sentence)) return false;
  if (lex.studyContext.test(sentence)) return true;
  return !lex.selfReferential.test(sentence.trim());
}

// Exported separately from extractUncitedClaims so the LLM-provider path
// (whose uncited claims come from the model, not this local rule) can still
// bucket its own results into the same three report groups for display.
export function categorizeUncitedClaim(sentence, lang = 'en') {
  const lex = LEXICON[lang] || LEXICON.en;
  if (NUMBER_PATTERN.test(sentence)) return 'STATISTIC';
  if (lex.comparative.test(sentence)) return 'COMPARATIVE';
  return 'INTERPRETIVE';
}

// Turns the raw `isLikelyClaim`-flagged, uncited candidates from
// `extractClaims` into the actual "no citation" review list — layering on
// the checks that keep the local rule from drowning a real issue in noise:
// discourse-marker sentences, sentences within one sentence of a citation in
// the same paragraph (citation-inheritance window), and sentences with no
// concrete empirical signal once discourse framing is stripped out.
export function extractUncitedClaims(allClaims, lang = 'en') {
  const out = [];
  allClaims.forEach((c, i) => {
    if (!c.autoSelected || c.hasCitation) return;
    if (startsWithDiscourseMarker(c.text, lang)) return;

    const prev = allClaims[i - 1];
    const next = allClaims[i + 1];
    const prevCites = prev && prev.paragraphIndex === c.paragraphIndex && prev.hasCitation;
    const nextCites = next && next.paragraphIndex === c.paragraphIndex && next.hasCitation;
    if (prevCites || nextCites) return;

    if (!hasConcreteEmpiricalSignal(c.text, lang)) return;

    out.push({
      claim: c.text,
      note: 'Flagged by local rules (number, comparison, or reasoning verb) — no citation attached.',
      category: categorizeUncitedClaim(c.text, lang),
    });
  });
  return out;
}
