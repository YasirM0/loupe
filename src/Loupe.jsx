import { useState, useRef, useEffect, useMemo } from "react";
import { Upload, BookOpen, FileText, Trash2, CheckCircle,
         AlertCircle, XCircle, Loader, Copy, Settings, X, ArrowLeft,
         Download, AlertTriangle, ShieldCheck, ExternalLink, ChevronDown, ChevronUp } from "lucide-react";
import { extractPdfText, extractDocxText } from "./lib/extractText.js";
import { chunkReferenceIntoSentences, extractClaims } from "./lib/textProcessing.js";
import { BM25Index } from "./lib/bm25.js";

// ── Design tokens ─────────────────────────────────────────────────────────────
const C = {
  bg:      '#070C18',
  panel:   '#0E1525',
  card:    '#111927',
  border:  '#1C2A40',
  teal:    '#0EB5AD',
  tealD:   '#0A8A84',
  tealBg:  '#051A19',
  text:    '#D6DFF5',
  muted:   '#5E6E91',
  faint:   '#2E3D55',
  green:   '#22c55e',  greenBg: '#0D2B14',
  amber:   '#f59e0b',  amberBg: '#1A1505',
  red:     '#ef4444',  redBg:   '#1A0D0D',
  red2:    '#dc2626',
};

const STATUS = {
  SUPPORTED:    { col: C.green,  bg: C.greenBg, label: 'Supported',    Icon: CheckCircle  },
  PARTIAL:      { col: C.amber,  bg: C.amberBg, label: 'Partial',      Icon: AlertCircle  },
  UNSUPPORTED:  { col: C.red,    bg: C.redBg,   label: 'No source',    Icon: XCircle      },
  CONTRADICTED: { col: C.red2,   bg: C.redBg,   label: 'Contradicted', Icon: XCircle      },
};

const CONFIDENCE = {
  HIGH:   C.green,
  MEDIUM: C.amber,
  LOW:    C.muted,
};

// ── Providers ─────────────────────────────────────────────────────────────────
// Ordered by how they're presented in Settings: recommended local-only path
// first, then self-hosted local AI, then bring-your-own-key providers (kept
// separate, tucked behind a toggle, since that's a meaningfully bigger ask
// than clicking one button). "kind" picks the request/response shape —
// everything except Anthropic speaks the OpenAI-compatible chat/completions
// format, only base URL & default model differ.
const PROVIDER_DEFAULTS = {
  browser:     { label: 'Local — no API key, no setup', kind: 'browser', baseUrl: '', model: '', keyRequired: false, supportsPdf: true, tier: 'recommended' },
  local:       { label: 'Local AI (Ollama, LM Studio…)', kind: 'openai', baseUrl: 'http://localhost:11434/v1', model: '', keyRequired: false, supportsPdf: true, tier: 'local' },
  anthropic:   { label: 'Anthropic (Claude)', kind: 'anthropic', baseUrl: 'https://api.anthropic.com/v1', model: 'claude-sonnet-5', keyRequired: true,  supportsPdf: true, tier: 'api', aliases: ['claude', 'anthropic', 'sonnet', 'opus'] },
  openai:      { label: 'OpenAI',             kind: 'openai',    baseUrl: 'https://api.openai.com/v1', model: 'gpt-4o', keyRequired: true, supportsPdf: true, tier: 'api', aliases: ['openai', 'gpt', 'chatgpt'] },
  google:      { label: 'Google (Gemini)',    kind: 'openai',    baseUrl: 'https://generativelanguage.googleapis.com/v1beta/openai', model: 'gemini-2.5-flash', keyRequired: true, supportsPdf: true, tier: 'api', aliases: ['google', 'gemini'] },
  groq:        { label: 'Groq',               kind: 'openai',    baseUrl: 'https://api.groq.com/openai/v1', model: 'llama-3.3-70b-versatile', keyRequired: true, supportsPdf: true, tier: 'api', aliases: ['groq'] },
  openrouter:  { label: 'OpenRouter',         kind: 'openai',    baseUrl: 'https://openrouter.ai/api/v1', model: 'openai/gpt-4o', keyRequired: true, supportsPdf: true, tier: 'api', aliases: ['openrouter', 'router'] },
  huggingface: { label: 'Hugging Face',       kind: 'openai',    baseUrl: 'https://router.huggingface.co/v1', model: 'meta-llama/Llama-3.3-70B-Instruct', keyRequired: true, supportsPdf: true, tier: 'api', aliases: ['huggingface', 'hugging face', 'hf'] },
  deepseek:    { label: 'DeepSeek',           kind: 'openai',    baseUrl: 'https://api.deepseek.com/v1', model: 'deepseek-chat', keyRequired: true, supportsPdf: true, tier: 'api', aliases: ['deepseek'] },
  custom:      { label: 'Custom API',         kind: 'openai',    baseUrl: '', model: '', keyRequired: true, supportsPdf: true, tier: 'api', aliases: [] },
};
const API_KEY_PROVIDERS = Object.entries(PROVIDER_DEFAULTS).filter(([, d]) => d.tier === 'api' && d !== PROVIDER_DEFAULTS.custom).map(([k]) => k);

// Typed provider name -> best-matching known providers, for the "type a
// name, we recognize it and fill the rest in" box. Deliberately forgiving
// (prefix match either direction, or substring) since a typo shouldn't
// dump someone into "fill in everything yourself" when we could have helped.
function matchApiProviders(input) {
  const q = input.trim().toLowerCase();
  if (!q) return [];
  return API_KEY_PROVIDERS
    .map(key => ({ key, ...PROVIDER_DEFAULTS[key] }))
    .filter(d => d.aliases.some(a => a.startsWith(q) || q.startsWith(a) || a.includes(q)))
    .slice(0, 4);
}

// `quality` is a real measured number — accuracy against a small hand-built
// test set (bench-testset.mjs / bench-run.mjs in the repo, reproducible),
// not an estimate. 6 retrieval cases and 14 reasoning cases is a small
// sample, not a rigorous benchmark, but it's real signal instead of a guess.
// bge-small-en-v1.5 was dropped: it scored 83% on our test, worse than both
// options below, while also being larger than MiniLM — no axis it wins on.
// A wider sweep of MiniLM/E5/BGE-small-class candidates (2026-08-22, see
// bench/run.mjs) found 3 more that tied the top score, included below so
// they can be compared on real documents rather than just this 6-case set.
// bge-base-en-v1.5 also tied that top score but was removed after real-world
// use: it's the largest/slowest option here (~105MB, ~4x the per-embedding
// time of the others) for no measured accuracy edge over them on an actual
// paper, not just this synthetic test.
//
// `size` is the actual download size of the quantized ONNX file each model
// loads by default (measured via HEAD request), not the larger fp32
// checkpoint size that was previously (incorrectly) shown here.
//
// `pooling`/`queryPrefix`/`passagePrefix` encode each model's documented
// embedding convention — asymmetric models (e5, arctic-embed) score much
// worse if queries and reference passages aren't prefixed as their model
// card specifies; the worker applies these automatically per model.
// Trimmed to the 3 candidates that survived a much larger adversarial
// stress test (a synthetic paper + 5 real reference documents, 1,899
// indexed sentences — see bench/run.mjs's 2026-08-22 sweep notes) — every
// embedding model tried tied at recall@5 there, so ranking instead comes
// from the official small RETRIEVAL_TESTSET below plus size/speed.
const EMBED_MODELS = [
  { id: 'Xenova/all-MiniLM-L12-v2', label: 'all-MiniLM-L12-v2', size: '~32MB', quality: 100, desc: 'Default — 100% on our retrieval test (6 cases), best top-1 precision on the larger stress test of the tied options' },
  { id: 'Xenova/all-MiniLM-L6-v2',  label: 'all-MiniLM-L6-v2',  size: '~22MB',  quality: 100, desc: '100% on our test — smallest and fastest option, pick this if download size or speed matters more than the small top-1 precision edge L12 has' },
  // query-only prefix + CLS pooling per Snowflake's documented convention.
  // Scored only 50% (3/6) on our official retrieval test despite tying
  // everything else on the larger stress-test corpus — a genuinely weaker
  // result on the test that actually discriminates between these models,
  // kept as the cross-vendor option rather than the primary pick.
  { id: 'Snowflake/snowflake-arctic-embed-xs', label: 'snowflake-arctic-embed-xs', size: '~22MB', quality: 50, pooling: 'cls', queryPrefix: 'Represent this sentence for searching relevant passages: ', desc: '50% on our test — same size class as the default, from a different vendor, but noticeably weaker on our own accuracy benchmark; a cross-vendor fallback, not a top pick' },
];
// deberta-v3-base-zeroshot-v2.0 briefly held this default slot: highest
// raw accuracy of anything tried (86%/72%), but measuring actual per-claim
// inference speed (not just model load time) found it took ~2973ms per
// classification versus ~670ms for the options below — a ~700MB download
// that's also ~4.4x slower in use, for a gap the guards below closed to
// within a few points anyway. Demoted to a documented trade-off rather than
// shipped, once speed was weighed alongside accuracy instead of separately.
//
// nli-deberta-v3-small was tried as a fast/small backup and rejected for a
// different reason: 36% on the official test below, missing nearly every
// contradiction case — a severe, pre-existing weakness this same benchmark
// had already caught once before. tasksource-nli and nli-deberta-v3-base
// both hold up on both benchmarks.
const NLI_MODELS = [
  { id: 'Xenova/deberta-v3-base-tasksource-nli', label: 'deberta-v3-base-tasksource-nli', size: '~233MB', quality: 86, desc: 'Default — 86% on our reasoning test (14 cases), 668ms/classification. Ties deberta-v3-base-zeroshot-v2.0\'s accuracy on our larger adversarial test after the guards below, at a third of the download and 4.4x the speed' },
  { id: 'Xenova/nli-deberta-v3-base', label: 'nli-deberta-v3-base', size: '~233MB', quality: 79, desc: '79% on our test, 661ms/classification — similar size and speed to the default, from a different (classic MNLI/SNLI) training lineage rather than tasksource\'s' },
];
const RETRIEVAL_METHODS = [
  { id: 'rerank',     label: 'BM25 then rerank', desc: 'Default — BM25 top-30, embeddings rerank to top-5' },
  { id: 'bm25',        label: 'BM25 only',        desc: 'No embedding model download, fastest, least precise' },
  { id: 'embeddings',  label: 'Embeddings only',  desc: 'Semantic search, handles vocabulary mismatch' },
];

const CHUNK_WORDS = 900;
// How many chunks' worth of requests run concurrently. Higher is faster
// wall-clock but more likely to trip a provider's rate limit (which the
// existing 429 retry logic absorbs, just with added latency) — 3 is a
// reasonable middle ground for typical API tiers, including free ones.
const CHUNK_CONCURRENCY = 3;

// ── File reading ──────────────────────────────────────────────────────────────
// `.text` is always the real extracted text (used by the local browser
// pipeline, for chunking, and now as the PDF fallback for every LLM provider
// except Anthropic — see filePart below).
async function readFile(file) {
  const ext = file.name.split('.').pop().toLowerCase();
  if (ext === 'docx') {
    const text = await extractDocxText(file);
    return { name: file.name, type: 'text', content: text, text };
  }
  if (ext === 'pdf') {
    const [ab, text] = await Promise.all([file.arrayBuffer(), extractPdfText(file)]);
    const u8  = new Uint8Array(ab);
    let bin   = '';
    for (let i = 0; i < u8.length; i += 8192)
      bin += String.fromCharCode(...u8.slice(i, i + 8192));
    return { name: file.name, type: 'pdf', b64: btoa(bin), text };
  }
  const text = await file.text();
  return { name: file.name, type: 'text', content: text, text };
}

// Anthropic reads PDF bytes natively (better on tables/figures/layout); every
// other provider gets the pdfjs-extracted text instead of being blocked
// outright — real PDF support everywhere, just not the same fidelity.
function filePart(doc, label = 'SOURCE', kind = 'anthropic') {
  if (doc.type === 'pdf' && kind === 'anthropic')
    return { type: 'document',
             source: { type: 'base64', media_type: 'application/pdf', data: doc.b64 } };
  const text = doc.type === 'pdf' ? doc.text : doc.content;
  return { type: 'text', text: `[${label}: ${doc.name}]\n${text}` };
}

// Marks the last source block as an Anthropic prompt-cache breakpoint, since
// chunking means the same source documents get resent on every single call —
// caching them keeps that from multiplying token cost by the chunk count.
function buildRefParts(refs, kind) {
  const parts = refs.map(doc => filePart(doc, 'SOURCE', kind));
  if (kind === 'anthropic' && parts.length)
    parts[parts.length - 1] = { ...parts[parts.length - 1], cache_control: { type: 'ephemeral' } };
  return parts;
}

function chunkText(text) {
  const paras = text.split(/\n\s*\n/).map(p => p.trim()).filter(Boolean);
  const chunks = [];
  let cur = [], curWords = 0;
  for (const p of paras) {
    const w = p.split(/\s+/).length;
    if (curWords && curWords + w > CHUNK_WORDS) { chunks.push(cur.join('\n\n')); cur = []; curWords = 0; }
    cur.push(p); curWords += w;
  }
  if (cur.length) chunks.push(cur.join('\n\n'));
  return chunks.length ? chunks : [text];
}

// ── Prompting ─────────────────────────────────────────────────────────────────
const ABSOLUTE_RULES = `ABSOLUTE RULES — violate none of these:

1. GROUND EVERY VERDICT IN THE TEXT. Every status (SUPPORTED, PARTIAL, UNSUPPORTED, CONTRADICTED) must be based on text literally present in the provided SOURCE documents. If you cannot point to a specific passage, the status is UNSUPPORTED. Never infer, extrapolate, or use your own training knowledge to fill gaps.
2. NO TRAINING KNOWLEDGE. Treat yourself as having zero prior knowledge of the subject. You know only what is in the SOURCE documents provided here. If a claim seems true based on what you know about the world but no source supports it, the status is UNSUPPORTED — not SUPPORTED.
3. CONTRADICTED REQUIRES EXPLICIT TEXT. Only mark CONTRADICTED if a source contains text that directly contradicts the claim. Never mark CONTRADICTED from absence of support or logical inference alone.
4. QUOTE THE EVIDENCE. For every cited claim, include the exact short phrase (max 15 words) from the source that led to your verdict. If you cannot find one, the status is UNSUPPORTED and evidence is "not found in sources". Never fabricate a quote.
5. NAME THE SOURCE FILE PRECISELY. Use the exact filename as given in the [SOURCE: ...] labels. Never write "the sources" or "one document" — name the specific file. List all of them if more than one supports a claim.
6. DO NOT SUMMARIZE OR PARAPHRASE THE PAPER. Only check claims — you are not explaining the paper.
7. UNKNOWN IS VALID. If a claim is about something the sources do not address at all, status is UNSUPPORTED with explanation "not addressed in any source". This is a useful, honest verdict, not a failure.
8. CITATIONS ARE NOT EVIDENCE. A citation in the paper (e.g. "Smith, 2022") is not itself proof of support. Only check whether the claim's actual content appears in the uploaded SOURCE documents.

SELF-CHECK BEFORE RESPONDING: for every cited-claim verdict, ask "can I point to a specific sentence in a specific file that justifies this?" If not, change the status to UNSUPPORTED. Never respond without applying this check.`;

function claimInstructions({ whole }) {
  const scope = whole ? 'PAPER TO VERIFY' : 'PAPER EXCERPT';
  const citedCap = whole
    ? 'Pick the 20 most important cited claims (prioritize claims central to the argument if there are more).'
    : 'Check every cited claim in this excerpt — it is short enough to cover exhaustively, do not cap or pick "the most important" ones.';
  return `You are a source verification assistant. Your only job is to check whether claims in a research paper are supported by the SOURCE documents provided.

${ABSOLUTE_RULES}

Read the ${scope} and split its claims into two groups.

CITED CLAIMS: claims that carry an explicit citation (e.g. "(Smith, 2020)", "[3]", "Smith (2020) found that..."). ${citedCap}

UNCITED CLAIMS: factual, empirical, or statistical statements with no citation attached at all. Do not verify these against the sources — surface them so the author can confirm each is their own analysis, a restatement of something cited nearby, or common knowledge, rather than something that needed a source. Skip connective sentences, methodology narration, and clear opinion/argument. List up to 12 of the most notable.

Return ONLY valid JSON, no markdown fences:
{"citedClaims":[{"claim":"exact claim, max 90 chars","citation":"as it appears in the text","status":"SUPPORTED|PARTIAL|UNSUPPORTED|CONTRADICTED","evidence":"exact quote from source, max 15 words, or 'not found in sources'","explanation":"one sentence, max 80 chars","source":"exact filename, or 'none'","confidence":"HIGH|MEDIUM|LOW"}],"uncitedClaims":[{"claim":"claim text max 90 chars","note":"max 70 chars"}]}

Your entire response must be that JSON object and nothing else. No preamble like "Here is the JSON:", no explanation before or after it, no markdown code fence. The first character of your response must be {`;
}

const CONTRADICTION_INSTRUCTIONS = `You are hunting specifically for contradictions. Ignore whether claims are supported — that is checked separately. Your only job here is adversarial: actively look for any place where a SOURCE document says something that conflicts with, or gives a different fact or number than, the paper.

${ABSOLUTE_RULES}

Return ONLY valid JSON, no markdown fences:
{"contradictions":[{"claim":"the paper's claim, max 90 chars","evidence":"exact contradicting quote from a source, max 15 words","source":"exact filename","explanation":"what conflicts, max 80 chars"}]}
If you find none, return {"contradictions":[]}.

Your entire response must be that JSON object and nothing else. No preamble, no explanation, no markdown code fence. The first character of your response must be {`;

// ── LLM call ──────────────────────────────────────────────────────────────────
const sleep = ms => new Promise(res => setTimeout(res, ms));

// Pulls the first complete top-level JSON object out of a model's raw text,
// tolerating preamble/commentary a weaker model adds despite being told not
// to (e.g. "Here's the JSON:\n{...}"). Brace-counted so nested objects and
// arrays inside the payload don't confuse it.
function extractJsonObject(raw) {
  const cleaned = raw.replace(/```json|```/g, '').trim();
  const start = cleaned.indexOf('{');
  if (start === -1) return cleaned;
  let depth = 0;
  for (let i = start; i < cleaned.length; i++) {
    if (cleaned[i] === '{') depth++;
    else if (cleaned[i] === '}') { depth--; if (depth === 0) return cleaned.slice(start, i + 1); }
  }
  return cleaned.slice(start);
}

const MAX_ATTEMPTS = 3;
const RETRY_DELAY_MS = 2500;

// Retries on both connection failures and malformed JSON — local/weaker
// models are far more prone to both under memory pressure or long prompts
// than large hosted ones, and a short pause plus another try clears most of
// them. Trades time for reliability on purpose: this app has to work with
// whatever model the person in front of it can actually run, not just the
// biggest hosted ones.
async function callLLM({ kind, apiKey, baseUrl, model, parts, instructions, onRetry }, attempt = 1) {
  let url, headers, body, extractText;

  if (kind === 'anthropic') {
    url = `${baseUrl.replace(/\/+$/, '')}/messages`;
    headers = {
      'Content-Type': 'application/json',
      'x-api-key': apiKey,
      'anthropic-version': '2023-06-01',
      'anthropic-dangerous-direct-browser-access': 'true',
    };
    body = { model, max_tokens: 2000, messages: [{ role: 'user', content: [...parts, { type: 'text', text: instructions }] }] };
    extractText = d => d.content.filter(b => b.type === 'text').map(b => b.text).join('');
  } else {
    const textBlocks = parts.filter(p => p.type === 'text').map(p => p.text).join('\n\n');
    url = `${baseUrl.replace(/\/+$/, '')}/chat/completions`;
    headers = { 'Content-Type': 'application/json' };
    if (apiKey) headers.Authorization = `Bearer ${apiKey}`;
    body = { model, messages: [{ role: 'user', content: `${textBlocks}\n\n${instructions}` }], temperature: 0, max_tokens: 2000 };
    extractText = d => d.choices?.[0]?.message?.content || '';
  }

  const retry = async errFactory => {
    if (attempt < MAX_ATTEMPTS) {
      onRetry?.(attempt, MAX_ATTEMPTS);
      await sleep(RETRY_DELAY_MS * attempt);
      return callLLM({ kind, apiKey, baseUrl, model, parts, instructions, onRetry }, attempt + 1);
    }
    throw errFactory();
  };

  let r;
  try {
    r = await fetch(url, { method: 'POST', headers, body: JSON.stringify(body) });
  } catch (e) {
    return retry(() => {
      const origin = typeof window !== 'undefined' ? window.location.origin : 'this app\'s URL';
      return new Error(
        `Could not reach ${url} after ${MAX_ATTEMPTS} attempts. The request never got a response, which ` +
        `almost always means CORS or connectivity, not something wrong with your paper or prompt. Checklist: ` +
        `is the server actually running at that address? If it's a local server (Ollama, LM Studio, vLLM, ` +
        `etc.), does its CORS config allow requests from this exact origin — ${origin}? For Ollama: restart ` +
        `it with OLLAMA_ORIGINS=${origin} ollama serve. If it kept working then failing, check your machine ` +
        `isn't running low on memory — local models can crash or hang under memory pressure. (Raw browser ` +
        `error: ${e.message})`
      );
    });
  }

  // 429/5xx are worth retrying (rate limits, transient server trouble under
  // load); 4xx like a bad key or bad model name will just fail identically
  // again, so surface those immediately instead of wasting two more tries.
  if (!r.ok) {
    const errBody = await r.text().catch(() => '');
    if (r.status === 429 || r.status >= 500) {
      return retry(() => new Error(`API ${r.status} after ${MAX_ATTEMPTS} attempts${errBody ? ': ' + errBody.slice(0, 200) : ''}`));
    }
    throw new Error(`API ${r.status}${errBody ? ': ' + errBody.slice(0, 200) : ''}`);
  }

  const d = await r.json();
  const raw = extractText(d);
  try {
    return JSON.parse(extractJsonObject(raw));
  } catch (e) {
    return retry(() => new Error(
      `The model didn't return valid JSON after ${MAX_ATTEMPTS} attempts. Smaller or local models can ` +
      `struggle to follow a strict "JSON only" instruction on a long prompt — a larger local model (7B+ ` +
      `instruct-tuned) is usually far more reliable, or try a hosted provider. What it actually sent back: ` +
      `"${raw.slice(0, 200)}"`
    ));
  }
}

function summarize(cited, uncited, contradictions) {
  const counts = { SUPPORTED: 0, PARTIAL: 0, UNSUPPORTED: 0, CONTRADICTED: 0 };
  cited.forEach(c => { if (counts[c.status] !== undefined) counts[c.status]++; });
  const total = cited.length;
  const score = total ? Math.round(100 * (counts.SUPPORTED + 0.5 * counts.PARTIAL) / total) : 0;
  const summary = total
    ? `${total} cited claim${total !== 1 ? 's' : ''} checked: ${counts.SUPPORTED} supported, ${counts.PARTIAL} partial, ${counts.UNSUPPORTED} unsupported${counts.CONTRADICTED ? `, ${counts.CONTRADICTED} contradicted` : ''}. ${uncited.length} claim${uncited.length !== 1 ? 's' : ''} had no citation${contradictions.length ? `, and the contradiction pass flagged ${contradictions.length}.` : '.'}`
    : 'No cited claims were found to check.';
  return { score, summary, counts };
}

function buildFinalResult(cited, uncited, contradictions) {
  return { citedClaims: cited, uncitedClaims: uncited, contradictions, ...summarize(cited, uncited, contradictions) };
}

// A raw Date.now() epoch (e.g. loupe-report-1787354121516.txt) is unreadable
// and gives no sense of when the file was made — this reads at a glance and
// still won't collide within the same minute in normal use.
function filenameTimestamp() {
  const d = new Date();
  const pad = n => String(n).padStart(2, '0');
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}_${pad(d.getHours())}${pad(d.getMinutes())}`;
}

// Strips the extension and anything not safe/pleasant in a filename, so a
// paper named "Beyond remittances: knowledge transfer (draft v2).docx"
// becomes a readable "Beyond-remittances-knowledge-transfer-draft-v2".
function slugifyFilename(name) {
  return name.replace(/\.[^.]+$/, '').replace(/[^a-zA-Z0-9]+/g, '-').replace(/^-+|-+$/g, '').slice(0, 50);
}

// ── Sub-components ────────────────────────────────────────────────────────────
function DropZone({ label, sub, multi, onFile, onFiles, flush, busy, compact }) {
  const ref = useRef();
  const [isDragging, setIsDragging] = useState(false);
  const takeFiles = async files => {
    if (!files.length) return;
    if (multi && onFiles) await onFiles(files);
    else if (onFile) await onFile(files[0]);
  };
  const onChange = async e => {
    await takeFiles([...e.target.files]);
    e.target.value = '';
  };
  const onDrop = async e => {
    e.preventDefault();
    setIsDragging(false);
    if (busy) return;
    await takeFiles([...e.dataTransfer.files]);
  };
  return (
    <div
      onClick={() => { if (!busy) ref.current?.click(); }}
      onMouseEnter={e => { if (!isDragging && !busy) e.currentTarget.style.background = C.panel; }}
      onMouseLeave={e => { if (!isDragging) e.currentTarget.style.background = flush ? 'transparent' : C.card; }}
      onDragOver={e => { if (!busy) { e.preventDefault(); setIsDragging(true); } }}
      onDragLeave={() => setIsDragging(false)}
      onDrop={onDrop}
      style={{
        border: isDragging ? `1px dashed ${C.teal}` : (flush ? 'none' : `1px solid ${C.border}`),
        borderRadius: flush ? 0 : 10,
        background: isDragging ? C.tealBg : (flush ? 'transparent' : C.card),
        padding: compact ? '16px 20px' : '44px 20px', textAlign: 'center',
        cursor: busy ? 'default' : 'pointer', transition: 'background .15s, border-color .15s',
        opacity: busy ? 0.7 : 1,
      }}
    >
      {compact ? (
        busy
          ? <div style={{ fontSize: 12.5, color: C.muted, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8 }}>
              <span style={{ width: 13, height: 13, borderRadius: '50%', border: `2px solid ${C.border}`, borderTop: `2px solid ${C.teal}`, animation: 'spin 0.8s linear infinite', display: 'inline-block' }} />
              Reading file{multi ? 's' : ''}…
            </div>
          : <span style={{ fontSize: 12.5, color: C.muted }}>+ {label}</span>
      ) : (
        <>
          <div style={{
            width: 64, height: 64, borderRadius: '50%', background: C.tealBg, border: `1px solid ${C.teal}33`,
            display: 'flex', alignItems: 'center', justifyContent: 'center', margin: '0 auto 16px',
          }}>
            {busy
              ? <span style={{ width: 24, height: 24, borderRadius: '50%', border: `2.5px solid ${C.teal}33`, borderTop: `2.5px solid ${C.teal}`, animation: 'spin 0.8s linear infinite', display: 'inline-block' }} />
              : <Upload size={28} color={C.teal} />}
          </div>
          <div style={{ fontSize: 14.5, color: C.text, fontWeight: 600 }}>{busy ? `Reading file${multi ? 's' : ''}…` : label}</div>
          {!busy && sub && <div style={{ fontSize: 11.5, color: C.muted, marginTop: 5 }}>{sub}</div>}
          {!busy && <div style={{ fontSize: 11.5, color: C.muted, marginTop: 5 }}>or drag and drop</div>}
        </>
      )}
      <input ref={ref} type="file" accept=".txt,.docx,.pdf" multiple={!!multi} onChange={onChange} style={{ display: 'none' }} />
    </div>
  );
}

function FileChip({ name, onRemove }) {
  return (
    <div style={{
      display: 'flex', alignItems: 'center', gap: 7, background: C.card,
      border: `1px solid ${C.border}`, borderRadius: 6, padding: '6px 12px',
      fontSize: 12.5, color: C.muted, maxWidth: 260,
    }}>
      <FileText size={11} style={{ flexShrink: 0 }} />
      <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{name}</span>
      {onRemove && (
        <button onClick={onRemove} style={{
          background: 'none', border: 'none', color: C.red, cursor: 'pointer',
          fontSize: 16, lineHeight: 1, padding: 0, marginLeft: 2,
        }}>×</button>
      )}
    </div>
  );
}

function ScoreSeal({ score }) {
  const col = score >= 75 ? C.green : score >= 50 ? C.amber : C.red;
  const size = 136, r = 58, circ = 2 * Math.PI * r, dash = (score / 100) * circ;
  return (
    <div style={{ position: 'relative', width: size, height: size, flexShrink: 0 }}>
      <svg width={size} height={size} style={{ transform: 'rotate(-90deg)' }}>
        <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke={C.border} strokeWidth={7} />
        <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke={col} strokeWidth={7}
                strokeDasharray={`${dash} ${circ}`} strokeLinecap="round"
                style={{ transition: 'stroke-dasharray 1s ease' }} />
      </svg>
      <div style={{ position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)', textAlign: 'center' }}>
        <div style={{ fontSize: 30, fontWeight: 800, color: col, lineHeight: 1 }}>{score}</div>
        <div style={{ fontSize: 10.5, color: C.muted, marginTop: 3 }}>/ 100</div>
      </div>
    </div>
  );
}

function ConfidenceTag({ level }) {
  if (!level || !CONFIDENCE[level]) return null;
  return (
    <span style={{
      fontSize: 9.5, fontWeight: 700, color: CONFIDENCE[level], border: `1px solid ${CONFIDENCE[level]}55`,
      borderRadius: 4, padding: '2px 6px', letterSpacing: '0.04em', textTransform: 'uppercase', marginLeft: 8,
    }}>
      {level}
    </span>
  );
}

function ClaimCard({ claim }) {
  const s = STATUS[claim.status] || STATUS.PARTIAL;
  return (
    <div style={{ background: C.card, border: `1px solid ${C.border}`, borderLeft: `3px solid ${s.col}`, borderRadius: 10, padding: '20px 24px', marginBottom: 14 }}>
      <div style={{ display: 'flex', gap: 16, alignItems: 'flex-start' }}>
        <span style={{
          background: s.bg, color: s.col, fontSize: 10.5, fontWeight: 700, padding: '4px 9px',
          borderRadius: 5, letterSpacing: '0.06em', textTransform: 'uppercase', whiteSpace: 'nowrap',
          flexShrink: 0, marginTop: 2, display: 'flex', alignItems: 'center',
        }}>
          {s.label}<ConfidenceTag level={claim.confidence} />
        </span>
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ fontSize: 15, color: C.text, lineHeight: 1.6, marginBottom: 8 }}>{claim.claim}</div>
          {claim.citation && <div style={{ fontSize: 12, color: C.muted, fontStyle: 'italic', marginBottom: 8 }}>{claim.citation}</div>}
          {claim.evidence && (
            <div style={{ fontSize: 13, color: C.muted, borderLeft: `2px solid ${C.faint}`, paddingLeft: 12, marginBottom: 8, lineHeight: 1.55 }}>
              "{claim.evidence}"
            </div>
          )}
          <div style={{ fontSize: 13, color: C.muted, lineHeight: 1.55 }}>{claim.explanation}</div>
          {claim.source && claim.source !== 'none' && (
            <div style={{ fontSize: 12, color: C.muted, marginTop: 8 }}>→ {claim.source}</div>
          )}
        </div>
      </div>
    </div>
  );
}

function UncitedClaimCard({ claim, suggestion, review, onMine, onHelp }) {
  return (
    <div style={{
      background: C.card, border: `1px dashed ${review === 'own' ? C.faint : C.border}`, borderRadius: 10,
      padding: '20px 24px', marginBottom: 14, opacity: review === 'own' ? 0.6 : 1,
    }}>
      <div style={{ display: 'flex', gap: 16, alignItems: 'flex-start' }}>
        <span style={{
          background: 'transparent', color: C.muted, border: `1px solid ${C.border}`, fontSize: 10.5, fontWeight: 700,
          padding: '4px 9px', borderRadius: 5, letterSpacing: '0.06em', textTransform: 'uppercase',
          whiteSpace: 'nowrap', flexShrink: 0, marginTop: 2,
        }}>
          {review === 'own' ? 'Marked as yours' : 'No citation'}
        </span>
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ fontSize: 15, color: C.text, lineHeight: 1.6, marginBottom: 6 }}>{claim.claim}</div>
          {claim.note && <div style={{ fontSize: 13, color: C.muted, lineHeight: 1.55, marginBottom: 10 }}>{claim.note}</div>}

          {review !== 'own' && (
            <div style={{ display: 'flex', gap: 10 }}>
              <button onClick={onMine} style={{ background: 'none', border: `1px solid ${C.border}`, borderRadius: 7, color: C.muted, cursor: 'pointer', fontSize: 11.5, padding: '6px 12px' }}>
                This is mine
              </button>
              {suggestion && (
                <button onClick={onHelp} style={{ background: 'none', border: `1px solid ${C.teal}55`, borderRadius: 7, color: C.teal, cursor: 'pointer', fontSize: 11.5, padding: '6px 12px' }}>
                  Help me cite this
                </button>
              )}
            </div>
          )}

          {review === 'shown' && suggestion && (
            <div style={{ marginTop: 12, borderLeft: `2px solid ${C.faint}`, paddingLeft: 12 }}>
              <div style={{ fontSize: 11, color: C.muted, textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 5 }}>
                Closest match in your sources — verify it yourself before citing
              </div>
              <div style={{ fontSize: 13, color: C.muted, lineHeight: 1.55, fontStyle: 'italic' }}>"{suggestion.text}"</div>
              <div style={{ fontSize: 12, color: C.muted, marginTop: 6 }}>→ {suggestion.source}</div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function FilterTabs({ result, active, onChange }) {
  const tabs = [
    { key: 'ALL', label: 'All Claims', count: result.citedClaims.length + result.uncitedClaims.length },
    { key: 'SUPPORTED', label: STATUS.SUPPORTED.label, count: result.counts.SUPPORTED },
    { key: 'PARTIAL', label: STATUS.PARTIAL.label, count: result.counts.PARTIAL },
    { key: 'UNSUPPORTED', label: STATUS.UNSUPPORTED.label, count: result.counts.UNSUPPORTED },
    { key: 'CONTRADICTED', label: STATUS.CONTRADICTED.label, count: result.counts.CONTRADICTED },
    { key: 'UNCITED', label: 'Uncited', count: result.uncitedClaims.length },
  ].filter(t => t.key === 'ALL' || t.count > 0);

  return (
    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginBottom: 28 }}>
      {tabs.map(t => (
        <button key={t.key} onClick={() => onChange(t.key)} style={{
          background: active === t.key ? C.teal : 'none',
          color: active === t.key ? '#070C18' : C.muted,
          border: `1px solid ${active === t.key ? C.teal : C.border}`,
          borderRadius: 20, padding: '7px 16px', fontSize: 12.5, fontWeight: active === t.key ? 700 : 500, cursor: 'pointer',
        }}>
          {t.label} ({t.count})
        </button>
      ))}
    </div>
  );
}

function ContradictionCard({ item }) {
  return (
    <div style={{ background: C.redBg, border: `1px solid ${C.red2}55`, borderLeft: `3px solid ${C.red2}`, borderRadius: 10, padding: '20px 24px', marginBottom: 14 }}>
      <div style={{ fontSize: 15, color: C.text, lineHeight: 1.6, marginBottom: 8 }}>{item.claim}</div>
      {item.evidence && (
        <div style={{ fontSize: 13, color: '#FCA5A5', borderLeft: `2px solid ${C.red2}`, paddingLeft: 12, marginBottom: 8, lineHeight: 1.55 }}>
          "{item.evidence}"
        </div>
      )}
      <div style={{ fontSize: 13, color: C.muted, lineHeight: 1.55 }}>{item.explanation}</div>
      {item.source && <div style={{ fontSize: 12, color: C.red, marginTop: 8 }}>→ {item.source}</div>}
    </div>
  );
}

function ProgressBar({ current, total }) {
  const pct = total ? Math.round((current / total) * 100) : 0;
  return (
    <div style={{ width: '100%', height: 6, background: C.card, borderRadius: 3, overflow: 'hidden' }}>
      <div style={{ width: `${pct}%`, height: '100%', background: C.teal, transition: 'width .3s ease' }} />
    </div>
  );
}

// Live stats while a run is in progress — fed by `liveClaims`, which grows
// as real results land (per-claim for the browser pipeline, per-batch for
// chunked LLM runs; stays empty for the single-call whole-document path,
// which has no intermediate signal to show). Reuses `summarize()` so the
// numbers here are computed the exact same way as the final result.
function LiveProgressPanel({ liveClaims }) {
  if (!liveClaims.length) return null;
  const { counts } = summarize(liveClaims, [], []);
  const issues = counts.UNSUPPORTED + counts.CONTRADICTED;
  const latest = liveClaims[liveClaims.length - 1];
  const stats = [
    { label: 'Checked', value: liveClaims.length, col: C.text },
    { label: 'Supported', value: counts.SUPPORTED, col: C.green },
    { label: 'Partial', value: counts.PARTIAL, col: C.amber },
    { label: 'Issues', value: issues, col: C.red },
  ];
  return (
    <div style={{ width: '100%', maxWidth: 420, display: 'flex', flexDirection: 'column', gap: 16 }}>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 10 }}>
        {stats.map(s => (
          <div key={s.label} style={{ background: C.card, border: `1px solid ${C.border}`, borderRadius: 8, padding: '12px 6px', textAlign: 'center' }}>
            <div style={{ fontSize: 20, fontWeight: 800, color: s.col, lineHeight: 1 }}>{s.value}</div>
            <div style={{ fontSize: 10.5, color: C.muted, marginTop: 5 }}>{s.label}</div>
          </div>
        ))}
      </div>
      {latest && (
        <div style={{ background: C.card, border: `1px solid ${C.border}`, borderRadius: 8, padding: '12px 16px' }}>
          <div style={{ fontSize: 10.5, color: C.muted, textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 6 }}>Latest checked</div>
          <div style={{ fontSize: 12.5, color: C.muted, lineHeight: 1.5, overflow: 'hidden', textOverflow: 'ellipsis', display: '-webkit-box', WebkitLineClamp: 2, WebkitBoxOrient: 'vertical' }}>
            {latest.claim}
          </div>
        </div>
      )}
    </div>
  );
}

function Footer() {
  return (
    <footer style={{ borderTop: `1px solid ${C.border}`, padding: '32px 24px 44px', textAlign: 'center' }}>
      <div style={{ fontSize: 12, color: C.muted, lineHeight: 1.75, maxWidth: 640, margin: '0 auto 16px' }}>
        Everything runs in your own browser. Your paper, sources, and API key are never sent to or stored on
        any server we run — we don't operate one. Your key goes straight from your browser to the AI provider
        you choose, and we never see or keep a copy of it. You're responsible for that provider's own usage,
        costs, and terms.
      </div>
      <div style={{ fontSize: 11.5, color: C.muted, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8 }}>
        <span>Built by</span>
        <a href="https://github.com/YasirM0" target="_blank" rel="noopener noreferrer" style={{ color: C.teal, textDecoration: 'none' }}>
          Yasir Mohammed
        </a>
        <span>·</span>
        <a href="https://github.com/YasirM0/loupe" target="_blank" rel="noopener noreferrer"
           style={{ color: C.teal, textDecoration: 'none', display: 'inline-flex', alignItems: 'center', gap: 4 }}>
          Source on GitHub <ExternalLink size={11} />
        </a>
      </div>
    </footer>
  );
}

function ProviderCard({ selected, title, badge, desc, onClick }) {
  return (
    <button onClick={onClick} style={{
      textAlign: 'left', display: 'flex', flexDirection: 'column', gap: 4,
      background: selected ? C.tealBg : C.card, border: `1px solid ${selected ? C.teal : C.border}`,
      borderRadius: 10, padding: '13px 15px', cursor: 'pointer',
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <span style={{ fontSize: 13.5, fontWeight: 600, color: selected ? C.teal : C.text }}>{title}</span>
        {badge && (
          <span style={{ fontSize: 9.5, fontWeight: 700, color: C.teal, border: `1px solid ${C.teal}55`, borderRadius: 4, padding: '2px 6px', letterSpacing: '0.04em', textTransform: 'uppercase' }}>
            {badge}
          </span>
        )}
      </div>
      <div style={{ fontSize: 11.5, color: C.muted, lineHeight: 1.55 }}>{desc}</div>
    </button>
  );
}

// The "type a name, we fill in the rest if we know it" box for bring-your-
// own-key providers. Typing shows matching known providers as clickable
// chips; clicking one (or typing a full name and moving on) fills base
// URL/model instantly. An unrecognized name just becomes a custom entry —
// someone typing their own provider's name presumably already has the
// base URL and model in hand.
function ApiKeySection({ provider, baseUrl, model, apiKey, huntContradictions, customProviderName,
                          onProvider, onBaseUrl, onModel, onApiKey, onToggleHunt, onCustomProviderName }) {
  const startingName = API_KEY_PROVIDERS.includes(provider)
    ? PROVIDER_DEFAULTS[provider].label
    : (provider === 'custom' ? customProviderName : '');
  const [nameInput, setNameInput] = useState(startingName);
  const [resolved, setResolved] = useState(!!startingName);
  const suggestions = resolved ? [] : matchApiProviders(nameInput);

  const commitMatch = key => {
    onProvider(key);
    setNameInput(PROVIDER_DEFAULTS[key].label);
    setResolved(true);
  };
  const commitCustom = () => {
    const name = nameInput.trim();
    if (!name) { setResolved(false); return; }
    if (suggestions.length === 1) { commitMatch(suggestions[0].key); return; }
    onProvider('custom');
    onCustomProviderName(name);
    setResolved(true);
  };

  return (
    <>
      <div>
        <div style={fieldLabelStyle}>Provider name</div>
        <input
          value={nameInput}
          onChange={e => { setNameInput(e.target.value); setResolved(false); }}
          onBlur={commitCustom}
          onKeyDown={e => { if (e.key === 'Enter') { e.preventDefault(); commitCustom(); } }}
          placeholder="e.g. claude, deepseek, openrouter…"
          style={inputStyle}
        />
        {!resolved && suggestions.length > 0 && (
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginTop: 8 }}>
            {suggestions.map(s => (
              <button key={s.key} onMouseDown={e => e.preventDefault()} onClick={() => commitMatch(s.key)} style={{
                background: C.card, border: `1px solid ${C.teal}55`, color: C.teal, borderRadius: 6,
                padding: '5px 10px', fontSize: 11.5, cursor: 'pointer',
              }}>
                {s.label}
              </button>
            ))}
          </div>
        )}
        {!resolved && nameInput.trim() && !suggestions.length && (
          <div style={{ fontSize: 11, color: C.muted, marginTop: 6, lineHeight: 1.5 }}>
            Not one we recognize — that's fine, click away and fill in the base URL and model yourself below.
          </div>
        )}
      </div>

      {resolved && (
        <>
          {PROVIDER_DEFAULTS[provider]?.kind !== 'anthropic' && (
            <div>
              <div style={fieldLabelStyle}>Base URL</div>
              <input value={baseUrl} onChange={e => onBaseUrl(e.target.value)} placeholder="e.g. https://api.example.com/v1" style={inputStyle} />
            </div>
          )}
          <div>
            <div style={fieldLabelStyle}>Model</div>
            <input value={model} onChange={e => onModel(e.target.value)} placeholder="Model name" style={inputStyle} />
          </div>
          <div>
            <div style={fieldLabelStyle}>API key</div>
            <input type="password" value={apiKey} onChange={e => onApiKey(e.target.value)} placeholder="API key" style={inputStyle} />
          </div>
          <label style={{ display: 'flex', alignItems: 'center', gap: 9, fontSize: 13, color: C.muted, cursor: 'pointer' }}>
            <input type="checkbox" checked={huntContradictions} onChange={onToggleHunt} />
            Hunt for contradictions (separate pass, ~2× calls)
          </label>
          <div style={{ fontSize: 11.5, color: C.muted, lineHeight: 1.65 }}>
            Stored only in this browser (localStorage), sent only to the base URL above.
          </div>
        </>
      )}
    </>
  );
}

function LocalAiFields({ baseUrl, model, apiKey, huntContradictions, onBaseUrl, onModel, onApiKey, onToggleHunt }) {
  return (
    <>
      <div>
        <div style={fieldLabelStyle}>Base URL</div>
        <input value={baseUrl} onChange={e => onBaseUrl(e.target.value)} placeholder="e.g. http://localhost:11434/v1" style={inputStyle} />
      </div>
      <div>
        <div style={fieldLabelStyle}>Model</div>
        <input value={model} onChange={e => onModel(e.target.value)} placeholder="Model name" style={inputStyle} />
      </div>
      <div>
        <div style={fieldLabelStyle}>API key</div>
        <input type="password" value={apiKey} onChange={e => onApiKey(e.target.value)} placeholder="Optional for local servers" style={inputStyle} />
      </div>
      <label style={{ display: 'flex', alignItems: 'center', gap: 9, fontSize: 13, color: C.muted, cursor: 'pointer' }}>
        <input type="checkbox" checked={huntContradictions} onChange={onToggleHunt} />
        Hunt for contradictions (separate pass, ~2× calls)
      </label>
      <div style={{ fontSize: 11.5, color: C.muted, lineHeight: 1.65 }}>
        Stored only in this browser (localStorage), sent only to the base URL above.
      </div>
    </>
  );
}

function SettingsModal({ provider, baseUrl, model, apiKey, huntContradictions, providerInfo,
                          embedModel, nliModel, retrievalMethod, customProviderName,
                          onProvider, onBaseUrl, onModel, onApiKey, onToggleHunt, onClose,
                          onEmbedModel, onNliModel, onRetrieval, onCustomProviderName }) {
  const [showApiProviders, setShowApiProviders] = useState(
    () => API_KEY_PROVIDERS.includes(provider) || provider === 'custom'
  );
  const [showAdvanced, setShowAdvanced] = useState(false);
  const selectTier = key => { onProvider(key); setShowApiProviders(false); };

  return (
    <div onClick={onClose} style={{
      position: 'fixed', inset: 0, background: 'rgba(4,7,14,0.72)', backdropFilter: 'blur(2px)',
      display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 50, padding: 20,
    }}>
      <div onClick={e => e.stopPropagation()} style={{
        background: C.panel, border: `1px solid ${C.border}`, borderRadius: 16,
        padding: '32px 34px', width: '100%', maxWidth: 460, maxHeight: '85vh', overflowY: 'auto',
        boxShadow: '0 24px 70px rgba(0,0,0,0.55)',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 26 }}>
          <div style={{ fontSize: 17, fontWeight: 700 }}>Connection</div>
          <button onClick={onClose} style={{ background: 'none', border: 'none', color: C.muted, cursor: 'pointer', padding: 4 }}>
            <X size={19} />
          </button>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          <div>
            <div style={fieldLabelStyle}>Provider</div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
              <ProviderCard
                selected={provider === 'browser' && !showApiProviders} badge="Recommended"
                title={PROVIDER_DEFAULTS.browser.label}
                desc="Runs fully in your browser. Your paper, sources, and results never leave your device — nothing is uploaded anywhere, no account needed, and it keeps working offline after the first load."
                onClick={() => selectTier('browser')}
              />
              <ProviderCard
                selected={provider === 'local' && !showApiProviders}
                title={PROVIDER_DEFAULTS.local.label}
                desc="Point at a model server running on your own machine (e.g. Ollama) for LLM-quality reasoning without an API key — needs that server running and a bit of setup."
                onClick={() => selectTier('local')}
              />
              <button onClick={() => setShowApiProviders(v => !v)} style={{
                background: 'none', border: 'none', color: C.teal, cursor: 'pointer',
                fontSize: 12, textAlign: 'left', padding: '4px 2px',
              }}>
                {showApiProviders ? 'Hide' : 'Or bring your own API key (Claude, DeepSeek, OpenRouter…)'}
              </button>
            </div>
          </div>

          {showApiProviders ? (
            <ApiKeySection
              provider={provider} baseUrl={baseUrl} model={model} apiKey={apiKey}
              huntContradictions={huntContradictions} customProviderName={customProviderName}
              onProvider={onProvider} onBaseUrl={onBaseUrl} onModel={onModel} onApiKey={onApiKey}
              onToggleHunt={onToggleHunt} onCustomProviderName={onCustomProviderName}
            />
          ) : provider === 'browser' ? (
            <>
              <div style={{ fontSize: 11.5, color: C.muted, lineHeight: 1.65, background: C.tealBg, border: `1px solid ${C.teal}22`, borderRadius: 8, padding: '10px 12px' }}>
                No API key, no account, no server. Two models download once to this browser and are cached — after that it works offline. Quality is more limited than an LLM-based provider (rule-based claim detection, embedding + NLI reasoning instead of full language understanding), but there's nothing to pay for and nothing to configure. The defaults are already the best-measured options below (real test results, not a guess) — most people don't need to open "Advanced."
              </div>
              <button onClick={() => setShowAdvanced(v => !v)} style={{
                background: 'none', border: 'none', color: C.teal, cursor: 'pointer',
                fontSize: 12, textAlign: 'left', padding: '4px 2px', display: 'flex', alignItems: 'center', gap: 5,
              }}>
                {showAdvanced ? <ChevronUp size={13} /> : <ChevronDown size={13} />} Advanced model settings
              </button>
              {showAdvanced && (
                <>
                  <div>
                    <div style={fieldLabelStyle}>Embedding model (retrieval)</div>
                    <select value={embedModel} onChange={e => onEmbedModel(e.target.value)} style={inputStyle}>
                      {EMBED_MODELS.map(m => <option key={m.id} value={m.id}>{m.label} — {m.size}{m.quality != null ? ` — ${m.quality}% on our test set` : ''}</option>)}
                    </select>
                    <div style={{ fontSize: 11, color: C.muted, marginTop: 4 }}>{EMBED_MODELS.find(m => m.id === embedModel)?.desc}</div>
                  </div>
                  <div>
                    <div style={fieldLabelStyle}>NLI model (reasoning)</div>
                    <select value={nliModel} onChange={e => onNliModel(e.target.value)} style={inputStyle}>
                      {NLI_MODELS.map(m => <option key={m.id} value={m.id}>{m.label} — {m.size}{m.quality != null ? ` — ${m.quality}% on our test set` : ''}</option>)}
                    </select>
                    <div style={{ fontSize: 11, color: C.muted, marginTop: 4 }}>{NLI_MODELS.find(m => m.id === nliModel)?.desc}</div>
                  </div>
                  <div>
                    <div style={fieldLabelStyle}>Retrieval method</div>
                    <select value={retrievalMethod} onChange={e => onRetrieval(e.target.value)} style={inputStyle}>
                      {RETRIEVAL_METHODS.map(m => <option key={m.id} value={m.id}>{m.label}</option>)}
                    </select>
                    <div style={{ fontSize: 11, color: C.muted, marginTop: 4 }}>{RETRIEVAL_METHODS.find(m => m.id === retrievalMethod)?.desc}</div>
                  </div>
                </>
              )}
            </>
          ) : (
            <LocalAiFields
              baseUrl={baseUrl} model={model} apiKey={apiKey} huntContradictions={huntContradictions}
              onBaseUrl={onBaseUrl} onModel={onModel} onApiKey={onApiKey} onToggleHunt={onToggleHunt}
            />
          )}
        </div>

        <button onClick={onClose} style={{
          width: '100%', marginTop: 28, background: C.teal, color: '#070C18', border: 'none',
          borderRadius: 9, padding: '14px', fontSize: 14.5, fontWeight: 700, cursor: 'pointer',
        }}>
          Done
        </button>
      </div>
    </div>
  );
}

// ── Main component ────────────────────────────────────────────────────────────
export default function Loupe() {
  const [provider, setProvider] = useState(() => localStorage.getItem('sv_provider') || 'browser');
  const [apiKey,   setApiKey]   = useState(() => localStorage.getItem('sv_api_key') || '');
  const [baseUrl,  setBaseUrl]  = useState(() => localStorage.getItem('sv_base_url') || PROVIDER_DEFAULTS.browser.baseUrl);
  const [model,    setModel]    = useState(() => localStorage.getItem('sv_model') || PROVIDER_DEFAULTS.browser.model);
  const [showSettings, setShowSettings] = useState(false);
  const [huntContradictions, setHuntContradictions] = useState(() => localStorage.getItem('sv_hunt') !== '0');
  const [embedModel, setEmbedModel] = useState(() => localStorage.getItem('sv_embed_model') || EMBED_MODELS[0].id);
  const [nliModel, setNliModel]     = useState(() => localStorage.getItem('sv_nli_model') || NLI_MODELS[0].id);
  const [retrievalMethod, setRetrievalMethod] = useState(() => localStorage.getItem('sv_retrieval') || RETRIEVAL_METHODS[0].id);
  const [customProviderName, setCustomProviderName] = useState(() => localStorage.getItem('sv_custom_name') || '');

  const [paper,    setPaper]    = useState(null);
  const [paperTxt, setPaperTxt] = useState('');
  const [pasteMode, setPasteMode] = useState(false);
  const [refs,     setRefs]     = useState([]);
  const [uploadingPaper, setUploadingPaper] = useState(false);
  const [uploadingRefs,  setUploadingRefs]  = useState(false); // PDF text extraction in particular can take real seconds — this is what tells the user it's working, not frozen
  // What "New verification" cleared, kept only so it can be restored —
  // covers reusing the same paper against different sources or vice versa,
  // without forcing a full re-upload of both sides every time.
  const [lastPaper,    setLastPaper]    = useState(null);
  const [lastPaperTxt, setLastPaperTxt] = useState('');
  const [lastRefs,     setLastRefs]     = useState([]);
  const [result,   setResult]   = useState(null);
  const [loading,  setLoading]  = useState(false);
  const [runStatus, setRunStatus] = useState('');
  const [liveChunk, setLiveChunk] = useState(null); // { current, total } for the active run only
  const [modelProgress, setModelProgress] = useState(null); // browser-pipeline model download progress
  const [liveClaims, setLiveClaims] = useState([]); // cited claims resolved so far in the active run — powers the live stats/activity panel
  const [activeFilter, setActiveFilter] = useState('ALL'); // results-screen filter tab
  const [uncitedReview, setUncitedReview] = useState({}); // { [uncitedClaimIndex]: 'own' | 'shown' } — per-claim "help me cite" state
  const [err,      setErr]      = useState(null);
  const [copied,   setCopied]   = useState(false);
  const [progress, setProgress] = useState(null); // paused/resumable snapshot, persisted
  const progressUpload = useRef();
  const workerRef = useRef(null);

  const providerInfo = PROVIDER_DEFAULTS[provider];

  useEffect(() => {
    try {
      const raw = localStorage.getItem('sv_progress');
      if (!raw) return;
      const snap = JSON.parse(raw);
      setProgress(snap);
      // A page reload wipes React state (paper/refs) even though this
      // snapshot survives in localStorage — without restoring them here,
      // clicking Resume would silently run with zero reference sources.
      setPaper(snap.paper || null);
      setPaperTxt(snap.paperTxt || snap.paper?.content || '');
      setRefs(snap.refs || []);
    } catch {}
  }, []);

  const handleProvider = p => {
    setProvider(p); localStorage.setItem('sv_provider', p);
    const d = PROVIDER_DEFAULTS[p];
    setBaseUrl(d.baseUrl); localStorage.setItem('sv_base_url', d.baseUrl);
    setModel(d.model);     localStorage.setItem('sv_model', d.model);
  };
  const handleApiKey = v => { setApiKey(v); if (v.trim()) localStorage.setItem('sv_api_key', v.trim()); else localStorage.removeItem('sv_api_key'); };
  const handleBaseUrl = v => { setBaseUrl(v); localStorage.setItem('sv_base_url', v); };
  const handleModel   = v => { setModel(v);   localStorage.setItem('sv_model', v); };
  const toggleHunt = () => { const v = !huntContradictions; setHuntContradictions(v); localStorage.setItem('sv_hunt', v ? '1' : '0'); };
  const handleEmbedModel = v => { setEmbedModel(v); localStorage.setItem('sv_embed_model', v); };
  const handleNliModel = v => { setNliModel(v); localStorage.setItem('sv_nli_model', v); };
  const handleRetrieval = v => { setRetrievalMethod(v); localStorage.setItem('sv_retrieval', v); };
  const handleCustomProviderName = v => { setCustomProviderName(v); localStorage.setItem('sv_custom_name', v); };

  const getWorker = () => {
    if (!workerRef.current) {
      workerRef.current = new Worker(new URL('./workers/inference.worker.js', import.meta.url), { type: 'module' });
    }
    return workerRef.current;
  };

  const handlePaper = async file => {
    setUploadingPaper(true);
    try { const d = await readFile(file); setPaper(d); setPaperTxt(d.content || ''); }
    catch (e) { setErr('Could not read paper: ' + e.message); }
    finally { setUploadingPaper(false); }
  };
  const handleRefs = async files => {
    setUploadingRefs(true);
    try { const loaded = await Promise.all(files.map(readFile)); setRefs(prev => [...prev, ...loaded]); }
    catch (e) { setErr('Could not read reference: ' + e.message); }
    finally { setUploadingRefs(false); }
  };

  const saveProgress = snap => {
    try { localStorage.setItem('sv_progress', JSON.stringify(snap)); } catch {}
    setProgress(snap);
  };
  const clearProgress = () => {
    try { localStorage.removeItem('sv_progress'); } catch {}
    setProgress(null);
  };

  const downloadProgress = () => {
    if (!progress) return;
    const blob = new Blob([JSON.stringify(progress, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = `loupe-progress-${filenameTimestamp()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };
  const uploadProgress = async file => {
    try {
      const snap = JSON.parse(await file.text());
      saveProgress(snap);
      setPaper(snap.paper || null);
      setPaperTxt(snap.paperTxt || snap.paper?.content || '');
      setRefs(snap.refs || []);
      setErr(null);
    } catch (e) { setErr('Could not read progress file: ' + e.message); }
  };

  // Shared by both a fresh run and a resume — resuming must not skip these.
  const validate = ({ forResume } = {}) => {
    if (providerInfo.kind !== 'browser') {
      if (providerInfo.keyRequired && !apiKey.trim()) return `Enter your ${providerInfo.label} API key.`;
      if (!baseUrl.trim()) return 'Enter the API base URL.';
      if (!model.trim()) return 'Enter a model name.';
    }
    if (!forResume) {
      const text = paperTxt.trim() || paper?.content || paper?.text || '';
      if (!text) return 'No paper text to verify.';
      if (!refs.length) return 'Upload at least one reference source.';
    }
    return null;
  };

  const runChunked = async resumeFrom => {
    const text = paperTxt.trim() || paper?.content || paper?.text || '';
    const chunks = resumeFrom?.chunks || chunkText(text);
    let cited = resumeFrom?.citedClaims || [];
    let uncited = resumeFrom?.uncitedClaims || [];
    let contradictions = resumeFrom?.contradictions || [];
    let i = resumeFrom?.chunkIndex || 0;

    const refParts = buildRefParts(refs, providerInfo.kind);
    setLoading(true); setErr(null); setResult(null);
    setActiveFilter('ALL'); setUncitedReview({});
    setLiveChunk({ current: i, total: chunks.length });
    setLiveClaims(cited);

    // Chunks within a batch run concurrently (independent requests, no
    // ordering dependency), and each chunk's own claim-check +
    // contradiction-hunt calls also run concurrently rather than back to
    // back. Progress still only saves at whole-batch boundaries, so
    // chunkIndex stays a simple linear pointer — resume logic is unchanged,
    // it just advances in bigger steps. If any request in a batch fails
    // (after its own internal retries), the whole batch is discarded and
    // resume restarts from the batch's first chunk — a small amount of
    // re-work, traded for not having to track partial-batch completion.
    while (i < chunks.length) {
      const batchEnd = Math.min(i + CHUNK_CONCURRENCY, chunks.length);
      const batchLabel = batchEnd - i > 1 ? `chunks ${i + 1}–${batchEnd}` : `chunk ${i + 1}`;
      setRunStatus(`Checking ${batchLabel} of ${chunks.length}…`);

      let batchResults;
      try {
        batchResults = await Promise.all(
          Array.from({ length: batchEnd - i }, (_, k) => i + k).map(async idx => {
            const paperParts = [{ type: 'text', text: `[PAPER EXCERPT ${idx + 1}/${chunks.length}]\n${chunks[idx]}` }];
            const retryStatus = (attempt, max) =>
              setRunStatus(`Checking ${batchLabel} of ${chunks.length} — a response wasn't usable, retrying (${attempt}/${max - 1})…`);
            const claimCall = callLLM({
              kind: providerInfo.kind, apiKey: apiKey.trim(), baseUrl: baseUrl.trim(), model: model.trim(),
              parts: [...refParts, ...paperParts], instructions: claimInstructions({ whole: false }),
              onRetry: retryStatus,
            });
            const contraCall = huntContradictions
              ? callLLM({
                  kind: providerInfo.kind, apiKey: apiKey.trim(), baseUrl: baseUrl.trim(), model: model.trim(),
                  parts: [...refParts, ...paperParts], instructions: CONTRADICTION_INSTRUCTIONS,
                  onRetry: retryStatus,
                })
              : Promise.resolve({ contradictions: [] });
            const [verdict, cv] = await Promise.all([claimCall, contraCall]);
            return { verdict, cv };
          })
        );
      } catch (e) {
        const snap = { chunks, chunkIndex: i, citedClaims: cited, uncitedClaims: uncited, contradictions, paper, paperTxt: text, refs };
        saveProgress(snap);
        setLoading(false);
        setLiveChunk(null);
        setErr(`Stopped around ${batchLabel} of ${chunks.length}: ${e.message}. Progress is saved — click Resume to continue, or download it below.`);
        return;
      }

      for (const { verdict, cv } of batchResults) {
        cited = cited.concat(verdict.citedClaims || []);
        uncited = uncited.concat(verdict.uncitedClaims || []);
        contradictions = contradictions.concat(cv.contradictions || []);
      }

      i = batchEnd;
      setLiveChunk({ current: i, total: chunks.length });
      setLiveClaims(cited);
      saveProgress({ chunks, chunkIndex: i, citedClaims: cited, uncitedClaims: uncited, contradictions, paper, paperTxt: text, refs });
    }

    setResult(buildFinalResult(cited, uncited, contradictions));
    clearProgress();
    setLoading(false);
    setLiveChunk(null);
    setRunStatus('');
  };

  const runWholeDocument = async () => {
    setLoading(true); setErr(null); setResult(null); setLiveClaims([]);
    setActiveFilter('ALL'); setUncitedReview({});
    setRunStatus(huntContradictions ? 'Checking claims and hunting for contradictions…' : 'Reading sources and checking claims…');
    const refParts = buildRefParts(refs, providerInfo.kind);
    const paperParts = [filePart(paper, 'PAPER', providerInfo.kind), { type: 'text', text: '[The document above is the PAPER TO VERIFY]' }];
    const retryStatus = (attempt, max) =>
      setRunStatus(`A response wasn't usable, retrying (${attempt}/${max - 1})…`);
    try {
      // The claim-check and contradiction-hunt calls are independent, so
      // they run concurrently instead of one after the other.
      const claimCall = callLLM({
        kind: providerInfo.kind, apiKey: apiKey.trim(), baseUrl: baseUrl.trim(), model: model.trim(),
        parts: [...refParts, ...paperParts], instructions: claimInstructions({ whole: true }),
        onRetry: retryStatus,
      });
      const contraCall = huntContradictions
        ? callLLM({
            kind: providerInfo.kind, apiKey: apiKey.trim(), baseUrl: baseUrl.trim(), model: model.trim(),
            parts: [...refParts, ...paperParts], instructions: CONTRADICTION_INSTRUCTIONS,
            onRetry: retryStatus,
          })
        : Promise.resolve({ contradictions: [] });
      const [verdict, cv] = await Promise.all([claimCall, contraCall]);
      setResult(buildFinalResult(verdict.citedClaims || [], verdict.uncitedClaims || [], cv.contradictions || []));
    } catch (e) { setErr('Verification failed: ' + e.message); }
    finally { setLoading(false); setRunStatus(''); }
  };

  // Runs entirely client-side: BM25 + embedding rerank to find the most
  // relevant reference sentences per claim, then an NLI model decides
  // support/contradiction/neutral. No network calls once the two models are
  // cached — this is the zero-API-key, zero-account path.
  const runBrowserVerification = async () => {
    setLoading(true); setErr(null); setResult(null); setModelProgress({}); setLiveClaims([]);
    setActiveFilter('ALL'); setUncitedReview({});
    const worker = getWorker();
    const paperText = paper?.text || paperTxt.trim();
    const allClaims = extractClaims(paperText);
    const citedCandidates = allClaims.filter(c => c.hasCitation);
    const uncitedClaims = allClaims
      .filter(c => c.autoSelected && !c.hasCitation)
      .map(c => ({ claim: c.text, note: 'Flagged by local rules (number, comparison, or reasoning verb) — no citation attached.' }));

    const onMessage = (resolveType, onMsg) => new Promise((resolve, reject) => {
      const handler = e => {
        const msg = e.data;
        if (msg.type === 'ERROR') { worker.removeEventListener('message', handler); reject(new Error(msg.message)); return; }
        if (msg.type === resolveType) { worker.removeEventListener('message', handler); resolve(msg); return; }
        onMsg?.(msg);
      };
      worker.addEventListener('message', handler);
    });

    try {
      setRunStatus('Loading local models (first run downloads them, then they\'re cached)…');
      const modelsReady = onMessage('MODELS_READY', msg => {
        if (msg.type === 'MODEL_PROGRESS') setModelProgress(prev => ({ ...prev, [msg.model]: msg }));
      });
      const nliDtype = NLI_MODELS.find(m => m.id === nliModel)?.dtype;
      const embedCfg = EMBED_MODELS.find(m => m.id === embedModel) || {};
      worker.postMessage({
        type: 'LOAD_MODELS', embedModel, nliModel, retrievalMethod, nliDtype,
        embedPooling: embedCfg.pooling || 'mean',
        embedQueryPrefix: embedCfg.queryPrefix || '',
        embedPassagePrefix: embedCfg.passagePrefix || '',
      });
      await modelsReady;
      setModelProgress(null);

      const refSentences = refs.flatMap(r => chunkReferenceIntoSentences(r.name, r.text || r.content || ''));
      setRunStatus(`Indexing ${refSentences.length} sentences from ${refs.length} source${refs.length !== 1 ? 's' : ''}…`);
      const indexed = onMessage('INDEXED', msg => {
        if (msg.type === 'INDEX_PROGRESS') setRunStatus(`Indexing sentence ${msg.done} of ${msg.total}…`);
      });
      worker.postMessage({ type: 'INDEX', sentences: refSentences, retrievalMethod });
      await indexed;

      setRunStatus(`Checking ${citedCandidates.length} cited claim${citedCandidates.length !== 1 ? 's' : ''}…`);
      const citedClaims = [];
      const contradictions = [];
      let done = 0;
      const verified = onMessage('VERIFY_DONE', msg => {
        if (msg.type !== 'CLAIM_RESULT') return;
        done++;
        setRunStatus(`Checking claim ${done} of ${citedCandidates.length}…`);
        const claim = citedCandidates.find(c => c.index === msg.claimIndex);
        const r = msg.result;
        const explanation = r.nliScores
          ? `Cosine ${r.cosineSim.toFixed(2)}, NLI ${Math.round(Math.max(r.nliScores.supported, r.nliScores.contradicted) * 100)}%`
          : 'No sufficiently similar passage found in sources.';
        const entry = { claim: claim.text, citation: claim.citationText || '', status: r.status, evidence: r.evidence, source: r.source, confidence: r.confidence, explanation };
        citedClaims.push(entry);
        setLiveClaims([...citedClaims]);
        if (r.status === 'CONTRADICTED') contradictions.push({ claim: claim.text, evidence: r.evidence, source: r.source, explanation });
      });
      worker.postMessage({ type: 'VERIFY', claims: citedCandidates, retrievalMethod });
      await verified;

      setResult(buildFinalResult(citedClaims, uncitedClaims, contradictions));
    } catch (e) {
      setErr('Verification failed: ' + e.message);
    } finally {
      setLoading(false); setRunStatus(''); setModelProgress(null);
    }
  };

  const run = () => {
    const problem = validate();
    if (problem) { setErr(problem); return; }
    if (providerInfo.kind === 'browser') runBrowserVerification();
    // Only Anthropic reads a PDF's raw bytes natively, so only it needs the
    // single whole-document pass — every other provider now has real
    // extracted text for a PDF paper too, so it can chunk it like any other.
    else if (paper?.type === 'pdf' && providerInfo.kind === 'anthropic') runWholeDocument();
    else runChunked();
  };

  // Actually starts fresh — stashes whatever was loaded so it can be
  // restored selectively, clears the setup, and discards any stale paused-
  // run banner (a leftover snapshot from an earlier, unrelated run has no
  // business surviving into a deliberately new verification).
  const startNewVerification = () => {
    setLastPaper(paper); setLastPaperTxt(paperTxt); setLastRefs(refs);
    setPaper(null); setPaperTxt(''); setRefs([]);
    setResult(null); setErr(null);
    setActiveFilter('ALL'); setUncitedReview({});
    clearProgress();
  };
  const restorePaper = () => { setPaper(lastPaper); setPaperTxt(lastPaperTxt); };
  const restoreRefs  = () => { setRefs(lastRefs); };

  const resume = () => {
    if (!progress) return;
    // This paused snapshot only ever comes from the LLM-provider chunked
    // path — the local browser pipeline has no resumability of its own.
    // Without this check, resuming while "Local — no API key, no setup" is
    // active would silently try an LLM-style request with an empty base
    // URL and fail confusingly far from the button that triggered it.
    if (providerInfo.kind === 'browser') {
      setErr('This paused run is from an AI-provider verification, not the local browser pipeline — switch to the provider you were using (via the settings icon) and click Resume again.');
      return;
    }
    const problem = validate({ forResume: true });
    if (problem) { setErr(problem); return; }
    runChunked(progress);
  };

  const buildReportText = () => {
    if (!result) return '';
    const lines = [
      `LOUPE — SOURCE VERIFICATION REPORT`, `Score: ${result.score}/100`, ``, result.summary, ``,
    ];
    if (result.contradictions.length) {
      lines.push(`CONTRADICTIONS`);
      lines.push(...result.contradictions.map(c => `[CONTRADICTED] ${c.claim}\n  "${c.evidence}"\n  ${c.explanation}\n  Source: ${c.source}`));
      lines.push('');
    }
    lines.push(`CITED CLAIMS`);
    lines.push(...result.citedClaims.map(c =>
      `[${c.status}] ${c.claim}${c.citation ? ' ' + c.citation : ''}\n  Evidence: "${c.evidence}"\n  ${c.explanation}${c.source && c.source !== 'none' ? '\n  Source: ' + c.source : ''}`
    ));
    lines.push('', `CLAIMS WITHOUT A CITATION`);
    lines.push(...result.uncitedClaims.map(c => `- ${c.claim}${c.note ? '\n  ' + c.note : ''}`));
    return lines.join('\n');
  };

  const copyReport = () => {
    if (!result) return;
    navigator.clipboard.writeText(buildReportText());
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const downloadReport = () => {
    if (!result) return;
    const blob = new Blob([buildReportText()], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    const nameHint = paper?.name ? slugifyFilename(paper.name) + '-' : '';
    a.href = url; a.download = `loupe-report-${nameHint}${filenameTimestamp()}.txt`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const paperText = paperTxt.trim() || paper?.content || paper?.text || '';
  const hasPaper  = !!paperText;
  const canRun    = (!providerInfo.keyRequired || !!apiKey.trim()) && hasPaper && refs.length > 0 && !loading;
  // Chunkable for every provider now except Anthropic+PDF, which stays a
  // single native-PDF pass (see the run() dispatcher for why).
  const singlePassPdf = paper?.type === 'pdf' && providerInfo.kind === 'anthropic';
  const isChunkable = providerInfo.kind !== 'browser' && !singlePassPdf && paperText;
  const estChunks = isChunkable ? chunkText(paperText).length : 1;
  const browserPaperText = paper?.text || paperTxt.trim();
  const browserClaimEstimate = providerInfo.kind === 'browser' && browserPaperText
    ? extractClaims(browserPaperText).filter(c => c.hasCitation).length
    : 0;
  const estCalls  = estChunks * (huntContradictions ? 2 : 1);
  const isPaused  = progress && progress.chunkIndex < progress.chunks?.length;

  // "Help me cite this" suggestions for uncited claims — plain BM25 lexical
  // search over the user's own uploaded refs, no embedding model or NLI
  // judgment involved (works for every provider, not just the browser
  // pipeline, since it only needs `refs`, already in state regardless of
  // which verification path ran). Recomputed only when the result or refs
  // change, not on every render.
  const uncitedSuggestions = useMemo(() => {
    if (!result?.uncitedClaims?.length || !refs.length) return [];
    const sentences = refs.flatMap(r => chunkReferenceIntoSentences(r.name, r.text || r.content || ''));
    if (!sentences.length) return [];
    const index = new BM25Index(sentences);
    return result.uncitedClaims.map(c => {
      const [top] = index.search(c.claim, 1);
      return top ? { text: top.doc.plainText, source: top.doc.sourceFile } : null;
    });
  }, [result, refs]);

  return (
    <div style={{ minHeight: '100vh', background: C.bg, color: C.text, fontFamily: 'Inter, system-ui, sans-serif' }}>

      {/* Header */}
      <div style={{
        position: 'sticky', top: 0, zIndex: 10, background: C.panel, borderBottom: `1px solid ${C.border}`,
        padding: '18px 36px', display: 'flex', alignItems: 'center', gap: 16, flexWrap: 'wrap',
      }}>
        <div style={{ width: 34, height: 34, borderRadius: 9, background: C.tealBg, border: `1px solid ${C.teal}33`, display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
          <BookOpen size={17} color={C.teal} />
        </div>
        <div style={{ minWidth: 0 }}>
          <div style={{ fontSize: 16, fontWeight: 700, letterSpacing: '-0.02em' }}>Loupe</div>
          <div style={{ fontSize: 11.5, color: C.muted, marginTop: 2 }}>
            Checks every citation, hunts for contradictions · no internet search
          </div>
        </div>

        <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 10 }}>
          {result && (
            <button onClick={startNewVerification} style={ghostBtnStyle}>
              <ArrowLeft size={13} /> New verification
            </button>
          )}
          {result && (
            <button onClick={copyReport} style={ghostBtnStyle}>
              <Copy size={13} /> {copied ? 'Copied!' : 'Copy report'}
            </button>
          )}
          {result && (
            <button onClick={downloadReport} style={ghostBtnStyle}>
              <Download size={13} /> Download report
            </button>
          )}
          <button onClick={() => setShowSettings(true)} title="Change what checks your claims — model, provider, or API key" style={ghostBtnStyle}>
            <Settings size={14} />
            <span>
              <span style={{ color: C.muted, fontWeight: 400 }}>Checking with: </span>
              {provider === 'custom' && customProviderName ? customProviderName : providerInfo.label}
            </span>
          </button>
        </div>
      </div>

      {showSettings && (
        <SettingsModal
          provider={provider} baseUrl={baseUrl} model={model} apiKey={apiKey}
          huntContradictions={huntContradictions} providerInfo={providerInfo}
          embedModel={embedModel} nliModel={nliModel} retrievalMethod={retrievalMethod}
          customProviderName={customProviderName}
          onProvider={handleProvider} onBaseUrl={handleBaseUrl} onModel={handleModel}
          onApiKey={handleApiKey} onToggleHunt={toggleHunt}
          onEmbedModel={handleEmbedModel} onNliModel={handleNliModel} onRetrieval={handleRetrieval}
          onCustomProviderName={handleCustomProviderName}
          onClose={() => setShowSettings(false)}
        />
      )}

      {/* Main content */}
      <div className="main-area">

        {!result && !loading && (
          <div className="setup">
            <div style={{ textAlign: 'center' }}>
              <h1 style={{ fontSize: 30, fontWeight: 800, letterSpacing: '-0.02em', margin: 0 }}>Verify your paper's claims</h1>
              <p style={{ fontSize: 13, color: C.muted, marginTop: 8, lineHeight: 1.55, maxWidth: 480, marginLeft: 'auto', marginRight: 'auto' }}>
                Checks every citation against your sources and flags anything stated without one.
              </p>
              <div style={{
                display: 'inline-flex', alignItems: 'center', gap: 7, marginTop: 16, fontSize: 11.5, color: C.muted,
                background: C.tealBg, border: `1px solid ${C.teal}22`, borderRadius: 20, padding: '7px 15px',
              }}>
                <ShieldCheck size={13} color={C.teal} /> Runs entirely in your browser — nothing is stored on any server
              </div>
            </div>

            {isPaused && (
              <div style={{ background: C.amberBg, border: `1px solid ${C.amber}44`, borderRadius: 10, padding: '16px 20px', display: 'flex', alignItems: 'center', gap: 16, flexWrap: 'wrap' }}>
                <AlertTriangle size={16} color="#FBBF24" style={{ flexShrink: 0 }} />
                <div style={{ fontSize: 13, color: '#FBBF24', flex: 1, minWidth: 200 }}>
                  Paused at chunk {progress.chunkIndex} of {progress.chunks.length} from a previous run
                </div>
                <div style={{ display: 'flex', gap: 10 }}>
                  <button onClick={resume} style={smallBtnStyle(C.amber)}>Resume</button>
                  <button onClick={downloadProgress} style={{ ...smallBtnStyle(C.border), display: 'flex', alignItems: 'center', gap: 5 }}>
                    <Download size={12} /> Download
                  </button>
                </div>
              </div>
            )}

            <div className="setup-grid">
              <div>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                  <div style={labelStyle}>Paper to verify <span style={{ color: C.red, fontWeight: 400, textTransform: 'none', letterSpacing: 0 }}>required</span></div>
                  {!paper && !paperTxt.trim() && (lastPaper || lastPaperTxt.trim()) && (
                    <button onClick={restorePaper} style={{ background: 'none', border: 'none', color: C.teal, cursor: 'pointer', fontSize: 11.5, marginBottom: 12 }}>
                      Restore previous paper
                    </button>
                  )}
                </div>
                <div style={{ background: C.card, border: `1px solid ${paper ? C.green + '44' : C.border}`, borderRadius: 12, overflow: 'hidden' }}>
                  {paper ? (
                    <div style={{ padding: '18px 20px', display: 'flex', alignItems: 'center', gap: 12 }}>
                      <FileText size={18} color={C.green} />
                      <div style={{ flex: 1, overflow: 'hidden' }}>
                        <div style={{ fontSize: 13.5, fontWeight: 600, color: C.text, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{paper.name}</div>
                        <div style={{ fontSize: 12, color: C.muted, marginTop: 3 }}>
                          {paper.type === 'pdf'
                            ? (singlePassPdf ? 'PDF document · single-pass (native, no chunking)' : 'PDF document · read as extracted text')
                            : `${(paper.content || '').split(' ').filter(Boolean).length} words`}
                        </div>
                      </div>
                      <button onClick={() => { setPaper(null); setPaperTxt(''); }} style={{ background: 'none', border: 'none', color: C.red, cursor: 'pointer' }}>
                        <Trash2 size={15} />
                      </button>
                    </div>
                  ) : (pasteMode || paperTxt.trim()) ? (
                    <>
                      <textarea value={paperTxt} onChange={e => setPaperTxt(e.target.value)} placeholder="Paste the paper text here…" autoFocus
                                style={{
                                  width: '100%', minHeight: paperTxt.trim() ? 340 : 160, background: 'transparent', border: 'none', outline: 'none',
                                  padding: '16px 20px', color: C.text, fontSize: 13, lineHeight: 1.65, resize: 'vertical', fontFamily: 'inherit',
                                  boxSizing: 'border-box', overflowWrap: 'break-word',
                                }} />
                      <div style={{ borderTop: `1px solid ${C.border}`, padding: '10px 20px' }}>
                        {paperTxt.trim() ? (
                          <button onClick={() => setPaperTxt('')} style={{ background: 'none', border: 'none', color: C.muted, cursor: 'pointer', fontSize: 12 }}>
                            Clear text
                          </button>
                        ) : (
                          <button onClick={() => setPasteMode(false)} style={{ background: 'none', border: 'none', color: C.muted, cursor: 'pointer', fontSize: 12 }}>
                            ← Back to file upload
                          </button>
                        )}
                      </div>
                    </>
                  ) : (
                    <>
                      <DropZone label="Upload paper" sub=".txt · .docx · .pdf" onFile={handlePaper} busy={uploadingPaper} flush />
                      <div style={{ borderTop: `1px solid ${C.border}`, padding: '10px 20px', textAlign: 'center' }}>
                        <button onClick={() => setPasteMode(true)} style={{ background: 'none', border: `1px solid ${C.border}`, borderRadius: 7, color: C.muted, cursor: 'pointer', fontSize: 12, padding: '7px 16px' }}>
                          Or paste text instead
                        </button>
                      </div>
                    </>
                  )}
                </div>
                {isChunkable && (
                  <div style={{ fontSize: 11.5, color: C.muted, marginTop: 8 }}>
                    ~{estChunks} chunk{estChunks !== 1 ? 's' : ''} · ~{estCalls} API call{estCalls !== 1 ? 's' : ''} to check every cited claim
                  </div>
                )}
                {providerInfo.kind === 'browser' && browserClaimEstimate > 0 && (
                  <div style={{ fontSize: 11.5, color: C.muted, marginTop: 8 }}>
                    ~{browserClaimEstimate} cited claim{browserClaimEstimate !== 1 ? 's' : ''} detected — no API calls, runs entirely on-device
                  </div>
                )}
              </div>

              <div>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                  <div style={labelStyle}>Reference sources</div>
                  {refs.length === 0 && lastRefs.length > 0 && (
                    <button onClick={restoreRefs} style={{ background: 'none', border: 'none', color: C.teal, cursor: 'pointer', fontSize: 11.5, marginBottom: 12 }}>
                      Restore previous sources
                    </button>
                  )}
                </div>
                <div style={{ background: C.card, border: `1px solid ${refs.length ? C.green + '44' : C.border}`, borderRadius: 12, overflow: 'hidden' }}>
                  {refs.length > 0 ? (
                    <div style={{ padding: '14px 16px 4px', display: 'flex', flexDirection: 'column', gap: 8, maxHeight: 240, overflowY: 'auto' }}>
                      {refs.map((r, i) => (
                        <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 10, background: C.panel, border: `1px solid ${C.border}`, borderRadius: 8, padding: '9px 12px' }}>
                          <FileText size={14} color={C.green} style={{ flexShrink: 0 }} />
                          <span style={{ flex: 1, fontSize: 12.5, color: C.text, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{r.name}</span>
                          <button onClick={() => setRefs(prev => prev.filter((_, j) => j !== i))} style={{ background: 'none', border: 'none', color: C.red, cursor: 'pointer', fontSize: 16, lineHeight: 1, padding: 0, flexShrink: 0 }}>×</button>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <DropZone label="Upload sources (multiple)" sub=".txt · .docx · .pdf" multi onFiles={handleRefs} busy={uploadingRefs} flush />
                  )}
                  <div style={{ borderTop: refs.length ? `1px solid ${C.border}` : 'none', marginTop: refs.length ? 10 : 0 }}>
                    {refs.length > 0 ? (
                      <DropZone label="Add more sources" multi onFiles={handleRefs} busy={uploadingRefs} flush compact />
                    ) : (
                      <div style={{ borderTop: `1px solid ${C.border}`, padding: '10px 20px', textAlign: 'center' }}>
                        <span style={{ display: 'inline-block', border: `1px solid transparent`, borderRadius: 7, color: C.muted, cursor: 'default', fontSize: 12, padding: '7px 16px' }}>
                          Add as many as you need
                        </span>
                      </div>
                    )}
                  </div>
                </div>
                {refs.length === 0 && (
                  <div style={{ fontSize: 12.5, color: C.muted, marginTop: 14, lineHeight: 1.6 }}>
                    The documents you want claims cross-checked against — papers, reports, datasets, anything with the facts your paper cites.
                  </div>
                )}
              </div>
            </div>

            <div style={{ textAlign: 'center' }}>
              <button onClick={run} disabled={!canRun} style={{
                background: canRun ? C.teal : C.card, color: canRun ? '#070C18' : C.muted,
                border: `1px solid ${canRun ? C.teal : C.border}`, borderRadius: 10, padding: '16px 40px',
                fontSize: 15, fontWeight: 700, cursor: canRun ? 'pointer' : 'not-allowed',
                display: 'inline-flex', alignItems: 'center', justifyContent: 'center', gap: 10, transition: 'all .15s',
              }}>
                <BookOpen size={17} /> Verify Claims
              </button>
              {err && (
                <div style={{ background: '#1A0D0D', border: `1px solid #5D1A1A`, borderRadius: 9, padding: '13px 18px', fontSize: 13, color: '#FCA5A5', lineHeight: 1.6, marginTop: 18, maxWidth: 560, marginLeft: 'auto', marginRight: 'auto' }}>
                  {err}
                </div>
              )}
              {!isPaused && (
                <div style={{ marginTop: 16 }}>
                  <button onClick={() => progressUpload.current?.click()} style={{
                    background: 'none', border: 'none', color: C.muted, cursor: 'pointer',
                    fontSize: 12, textDecoration: 'underline', textUnderlineOffset: 3,
                  }}>
                    Resuming on a different machine? Load a progress file
                  </button>
                </div>
              )}
            </div>

            <p style={{ fontSize: 12.5, color: C.muted, lineHeight: 1.7, textAlign: 'center', maxWidth: 620, margin: '0 auto' }}>
              Text/docx papers are split into chunks so every cited claim gets checked, not just the top few — PDF papers run in a single pass.
              A dedicated pass hunts for contradictions. If a run is interrupted, progress saves automatically so you can resume it.
            </p>
          </div>
        )}

        {loading && (
          <div style={{ minHeight: '50vh', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 20 }}>
            <div style={{ width: 52, height: 52, borderRadius: '50%', border: `3px solid ${C.border}`, borderTop: `3px solid ${C.teal}`, animation: 'spin 1s linear infinite' }} />
            <div style={{ fontSize: 14.5, color: C.muted }}>{runStatus || 'Reading sources and checking claims…'}</div>
            {liveChunk && <div style={{ width: 280 }}><ProgressBar current={liveChunk.current} total={liveChunk.total} /></div>}
            {modelProgress && Object.keys(modelProgress).length > 0 && (
              <div style={{ width: 280, display: 'flex', flexDirection: 'column', gap: 10 }}>
                {Object.entries(modelProgress).map(([key, p]) => (
                  <div key={key}>
                    <div style={{ fontSize: 11.5, color: C.muted, marginBottom: 4 }}>
                      {key === 'embedding' ? 'Embedding model' : 'NLI model'} — {typeof p.progress === 'number' ? `${Math.round(p.progress)}%` : (p.status || 'loading')}
                    </div>
                    <ProgressBar current={typeof p.progress === 'number' ? p.progress : 0} total={100} />
                  </div>
                ))}
              </div>
            )}
            <LiveProgressPanel liveClaims={liveClaims} />
          </div>
        )}

        {result && (
          <div className="results">
            <div style={{ background: C.panel, border: `1px solid ${C.border}`, borderRadius: 14, padding: '32px 36px', marginBottom: 36, display: 'flex', gap: 34, alignItems: 'center', flexWrap: 'wrap' }}>
              <ScoreSeal score={result.score} />
              <div style={{ flex: 1, minWidth: 240 }}>
                <div style={{ fontSize: 11, fontWeight: 700, color: C.muted, textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 10 }}>Citation Accuracy Score</div>
                <div style={{ fontSize: 15, color: C.text, lineHeight: 1.75 }}>{result.summary}</div>
                <div style={{ display: 'flex', gap: 20, marginTop: 18, flexWrap: 'wrap' }}>
                  {['SUPPORTED', 'PARTIAL', 'UNSUPPORTED', 'CONTRADICTED'].map(s => {
                    const st = STATUS[s], count = result.counts[s];
                    if (!count) return null;
                    return (
                      <div key={s} style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                        <div style={{ width: 8, height: 8, borderRadius: '50%', background: st.col }} />
                        <span style={{ fontSize: 12, color: C.muted }}>{count} {st.label.toLowerCase()}</span>
                      </div>
                    );
                  })}
                  {result.uncitedClaims.length > 0 && (
                    <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                      <div style={{ width: 8, height: 8, borderRadius: '50%', background: C.faint }} />
                      <span style={{ fontSize: 12, color: C.muted }}>{result.uncitedClaims.length} uncited</span>
                    </div>
                  )}
                </div>
              </div>
            </div>

            {result.contradictions.length > 0 && (
              <>
                <div style={{ fontSize: 12, fontWeight: 700, color: C.red2, textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 16, display: 'flex', alignItems: 'center', gap: 7 }}>
                  <AlertTriangle size={14} /> Contradictions Found
                </div>
                {result.contradictions.map((c, i) => <ContradictionCard key={i} item={c} />)}
              </>
            )}

            <FilterTabs result={result} active={activeFilter} onChange={setActiveFilter} />

            {activeFilter !== 'UNCITED' && (
              <>
                <div style={{ fontSize: 12, fontWeight: 700, color: C.muted, textTransform: 'uppercase', letterSpacing: '0.08em', marginTop: result.contradictions.length ? 36 : 0, marginBottom: 16 }}>
                  Cited Claims — Checked Against Sources
                </div>
                {result.citedClaims
                  .filter(c => activeFilter === 'ALL' || c.status === activeFilter)
                  .map((c, i) => <ClaimCard key={i} claim={c} />)}
              </>
            )}

            {result.uncitedClaims.length > 0 && (activeFilter === 'ALL' || activeFilter === 'UNCITED') && (
              <>
                <div style={{ fontSize: 12, fontWeight: 700, color: C.muted, textTransform: 'uppercase', letterSpacing: '0.08em', marginTop: 36, marginBottom: 10 }}>
                  Claims With No Citation
                </div>
                <div style={{ fontSize: 13, color: C.muted, lineHeight: 1.6, marginBottom: 16 }}>
                  Not checked against your sources — review these yourself to confirm they're your own analysis.
                </div>
                {result.uncitedClaims.map((c, i) => (
                  <UncitedClaimCard
                    key={i} claim={c} suggestion={uncitedSuggestions[i]}
                    review={uncitedReview[i]}
                    onMine={() => setUncitedReview(prev => ({ ...prev, [i]: 'own' }))}
                    onHelp={() => setUncitedReview(prev => ({ ...prev, [i]: 'shown' }))}
                  />
                ))}
              </>
            )}

            <div style={{ marginTop: 36, background: C.panel, border: `1px solid ${C.border}`, borderRadius: 10, padding: '20px 24px' }}>
              <div style={{ fontSize: 12, fontWeight: 700, color: C.muted, textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 14 }}>Sources checked</div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
                {refs.map((r, i) => <FileChip key={i} name={r.name} />)}
              </div>
            </div>
          </div>
        )}
      </div>

      <Footer />

      <input ref={progressUpload} type="file" accept=".json" style={{ display: 'none' }}
             onChange={e => { if (e.target.files[0]) uploadProgress(e.target.files[0]); e.target.value = ''; }} />

      <style>{`
        @keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: ${C.bg}; }
        ::-webkit-scrollbar-thumb { background: ${C.border}; border-radius: 3px; }
        textarea:focus, input:focus, select:focus { border-color: ${C.teal} !important; }

        .main-area { padding: 56px 32px 80px; overflow-x: hidden; }
        .setup { max-width: 920px; margin: 0 auto; display: flex; flex-direction: column; gap: 40px; }
        .setup-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 32px; }
        /* Grid items default to a min-width equal to their content's intrinsic
           width — one long unbroken string (a filename, a URL pasted into the
           paper text) can force a column, and the whole grid, wider than the
           viewport without this. */
        .setup-grid > div { min-width: 0; }
        .results { max-width: 880px; margin: 0 auto; }

        @media (max-width: 760px) {
          .main-area { padding: 32px 18px 56px; }
          .setup-grid { grid-template-columns: 1fr; }
          .setup { gap: 32px; }
        }
      `}</style>
    </div>
  );
}

const inputStyle = {
  width: '100%', background: C.card, border: `1px solid ${C.border}`, borderRadius: 8,
  padding: '12px 14px', color: C.text, fontSize: 13, fontFamily: 'inherit', boxSizing: 'border-box', outline: 'none',
};
const labelStyle = {
  fontSize: 11.5, fontWeight: 700, color: C.muted, textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 14,
};
const fieldLabelStyle = { fontSize: 11.5, fontWeight: 600, color: C.muted, marginBottom: 7 };
const smallBtnStyle = col => ({
  background: 'none', border: `1px solid ${col}`, color: C.text, borderRadius: 7,
  padding: '9px 14px', fontSize: 12.5, cursor: 'pointer',
});
const ghostBtnStyle = {
  background: 'none', border: `1px solid ${C.border}`, borderRadius: 8, padding: '9px 14px',
  color: C.muted, cursor: 'pointer', fontSize: 12.5, display: 'flex', alignItems: 'center', gap: 7,
};
