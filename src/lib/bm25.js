const STOPWORDS = new Set([
  'a','an','the','and','or','but','if','then','so','of','in','on','at','to','for','with',
  'by','from','as','is','are','was','were','be','been','being','it','its','this','that',
  'these','those','we','they','he','she','you','i','not','no','do','does','did','has','have','had',
]);

function tokenize(text) {
  return text.toLowerCase().split(/[^a-z0-9]+/).filter(t => t && !STOPWORDS.has(t));
}

// Standard Okapi BM25 over a fixed corpus of short documents (here: sentences).
// k1/b are the conventional defaults; no external library needed for this size.
export class BM25Index {
  constructor(docs, { k1 = 1.5, b = 0.75 } = {}) {
    this.docs = docs; // array of { text, ...meta }
    this.k1 = k1;
    this.b = b;
    this.tokenized = docs.map(d => tokenize(d.text));
    this.docLengths = this.tokenized.map(t => t.length);
    this.avgDocLength = this.docLengths.reduce((a, c) => a + c, 0) / (this.docLengths.length || 1);

    this.df = new Map(); // term -> number of docs containing it
    this.tokenized.forEach(tokens => {
      new Set(tokens).forEach(t => this.df.set(t, (this.df.get(t) || 0) + 1));
    });
    const N = docs.length;
    this.idf = new Map();
    for (const [term, freq] of this.df) {
      this.idf.set(term, Math.log(1 + (N - freq + 0.5) / (freq + 0.5)));
    }
  }

  score(queryTokens, docIndex) {
    const tokens = this.tokenized[docIndex];
    const termCounts = new Map();
    tokens.forEach(t => termCounts.set(t, (termCounts.get(t) || 0) + 1));
    const dl = this.docLengths[docIndex];
    let score = 0;
    for (const q of queryTokens) {
      const tf = termCounts.get(q) || 0;
      if (!tf) continue;
      const idf = this.idf.get(q) || 0;
      score += idf * (tf * (this.k1 + 1)) / (tf + this.k1 * (1 - this.b + this.b * dl / this.avgDocLength));
    }
    return score;
  }

  // Returns the top `n` { doc, score } pairs for a free-text query.
  search(query, n = 20) {
    const queryTokens = tokenize(query);
    if (!queryTokens.length) return [];
    const scored = this.docs.map((doc, i) => ({ doc, score: this.score(queryTokens, i) }));
    scored.sort((a, b) => b.score - a.score);
    return scored.filter(s => s.score > 0).slice(0, n);
  }
}

export function cosineSimilarity(a, b) {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  const denom = Math.sqrt(na) * Math.sqrt(nb);
  return denom ? dot / denom : 0;
}
