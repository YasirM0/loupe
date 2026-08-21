import { useState, useRef, useEffect } from "react";
import mammoth from "mammoth";
import { Upload, BookOpen, FileText, Trash2, CheckCircle,
         AlertCircle, XCircle, Loader, Copy, Settings, X, ArrowLeft,
         Download, AlertTriangle, ShieldCheck, ExternalLink } from "lucide-react";

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
// "kind" picks the request/response shape. Everything except Anthropic speaks
// the OpenAI-compatible chat/completions format (OpenAI itself, Google's and
// Hugging Face's OpenAI-compat routers, Groq, OpenRouter, and any local
// server — Ollama, LM Studio, vLLM) — only base URL & default model differ.
const PROVIDER_DEFAULTS = {
  anthropic:   { label: 'Anthropic (Claude)', kind: 'anthropic', baseUrl: 'https://api.anthropic.com/v1', model: 'claude-sonnet-5', keyRequired: true,  supportsPdf: true  },
  openai:      { label: 'OpenAI',             kind: 'openai',    baseUrl: 'https://api.openai.com/v1', model: 'gpt-4o', keyRequired: true, supportsPdf: false },
  google:      { label: 'Google (Gemini)',    kind: 'openai',    baseUrl: 'https://generativelanguage.googleapis.com/v1beta/openai', model: 'gemini-2.5-flash', keyRequired: true, supportsPdf: false },
  groq:        { label: 'Groq',               kind: 'openai',    baseUrl: 'https://api.groq.com/openai/v1', model: 'llama-3.3-70b-versatile', keyRequired: true, supportsPdf: false },
  openrouter:  { label: 'OpenRouter',         kind: 'openai',    baseUrl: 'https://openrouter.ai/api/v1', model: 'openai/gpt-4o', keyRequired: true, supportsPdf: false },
  huggingface: { label: 'Hugging Face',       kind: 'openai',    baseUrl: 'https://router.huggingface.co/v1', model: 'meta-llama/Llama-3.3-70B-Instruct', keyRequired: true, supportsPdf: false },
  local:       { label: 'Local / custom (OpenAI-compatible)', kind: 'openai', baseUrl: 'http://localhost:11434/v1', model: '', keyRequired: false, supportsPdf: false },
};

const CHUNK_WORDS = 900;

// ── File reading ──────────────────────────────────────────────────────────────
async function readFile(file) {
  const ext = file.name.split('.').pop().toLowerCase();
  if (ext === 'docx') {
    const ab  = await file.arrayBuffer();
    const res = await mammoth.extractRawText({ arrayBuffer: ab });
    return { name: file.name, type: 'text', content: res.value };
  }
  if (ext === 'pdf') {
    const ab  = await file.arrayBuffer();
    const u8  = new Uint8Array(ab);
    let bin   = '';
    for (let i = 0; i < u8.length; i += 8192)
      bin += String.fromCharCode(...u8.slice(i, i + 8192));
    return { name: file.name, type: 'pdf', b64: btoa(bin) };
  }
  return { name: file.name, type: 'text', content: await file.text() };
}

function filePart(doc, label = 'SOURCE') {
  if (doc.type === 'pdf')
    return { type: 'document',
             source: { type: 'base64', media_type: 'application/pdf', data: doc.b64 } };
  return { type: 'text', text: `[${label}: ${doc.name}]\n${doc.content}` };
}

// Marks the last source block as an Anthropic prompt-cache breakpoint, since
// chunking means the same source documents get resent on every single call —
// caching them keeps that from multiplying token cost by the chunk count.
function buildRefParts(refs, kind) {
  const parts = refs.map(doc => filePart(doc, 'SOURCE'));
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
{"citedClaims":[{"claim":"exact claim, max 90 chars","citation":"as it appears in the text","status":"SUPPORTED|PARTIAL|UNSUPPORTED|CONTRADICTED","evidence":"exact quote from source, max 15 words, or 'not found in sources'","explanation":"one sentence, max 80 chars","source":"exact filename, or 'none'","confidence":"HIGH|MEDIUM|LOW"}],"uncitedClaims":[{"claim":"claim text max 90 chars","note":"max 70 chars"}]}`;
}

const CONTRADICTION_INSTRUCTIONS = `You are hunting specifically for contradictions. Ignore whether claims are supported — that is checked separately. Your only job here is adversarial: actively look for any place where a SOURCE document says something that conflicts with, or gives a different fact or number than, the paper.

${ABSOLUTE_RULES}

Return ONLY valid JSON, no markdown fences:
{"contradictions":[{"claim":"the paper's claim, max 90 chars","evidence":"exact contradicting quote from a source, max 15 words","source":"exact filename","explanation":"what conflicts, max 80 chars"}]}
If you find none, return {"contradictions":[]}.`;

// ── LLM call ──────────────────────────────────────────────────────────────────
async function callLLM({ kind, apiKey, baseUrl, model, parts, instructions }) {
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
    body = { model, messages: [{ role: 'user', content: `${textBlocks}\n\n${instructions}` }], temperature: 0 };
    extractText = d => d.choices?.[0]?.message?.content || '';
  }

  const r = await fetch(url, { method: 'POST', headers, body: JSON.stringify(body) });
  if (!r.ok) {
    const errBody = await r.text().catch(() => '');
    throw new Error(`API ${r.status}${errBody ? ': ' + errBody.slice(0, 200) : ''}`);
  }
  const d = await r.json();
  const raw = extractText(d);
  return JSON.parse(raw.replace(/```json|```/g, '').trim());
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

// ── Sub-components ────────────────────────────────────────────────────────────
function DropZone({ label, sub, multi, onFile, onFiles, flush }) {
  const ref = useRef();
  const onChange = async e => {
    const files = [...e.target.files];
    if (!files.length) return;
    if (multi && onFiles) await onFiles(files);
    else if (onFile) await onFile(files[0]);
    e.target.value = '';
  };
  return (
    <div
      onClick={() => ref.current?.click()}
      onMouseEnter={e => e.currentTarget.style.background = C.panel}
      onMouseLeave={e => e.currentTarget.style.background = flush ? 'transparent' : C.card}
      style={{
        border: flush ? 'none' : `1px solid ${C.border}`, borderRadius: flush ? 0 : 10,
        background: flush ? 'transparent' : C.card,
        padding: '32px 20px', textAlign: 'center', cursor: 'pointer', transition: 'background .15s',
      }}
    >
      <div style={{
        width: 40, height: 40, borderRadius: '50%', background: C.tealBg, border: `1px solid ${C.teal}33`,
        display: 'flex', alignItems: 'center', justifyContent: 'center', margin: '0 auto 14px',
      }}>
        <Upload size={18} color={C.teal} />
      </div>
      <div style={{ fontSize: 13.5, color: C.text }}>{label}</div>
      {sub && <div style={{ fontSize: 11.5, color: C.faint, marginTop: 5 }}>{sub}</div>}
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
            <div style={{ fontSize: 12, color: C.teal, marginTop: 8 }}>→ {claim.source}</div>
          )}
        </div>
      </div>
    </div>
  );
}

function UncitedClaimCard({ claim }) {
  return (
    <div style={{ background: C.card, border: `1px dashed ${C.border}`, borderRadius: 10, padding: '20px 24px', marginBottom: 14 }}>
      <div style={{ display: 'flex', gap: 16, alignItems: 'flex-start' }}>
        <span style={{
          background: 'transparent', color: C.muted, border: `1px solid ${C.border}`, fontSize: 10.5, fontWeight: 700,
          padding: '4px 9px', borderRadius: 5, letterSpacing: '0.06em', textTransform: 'uppercase',
          whiteSpace: 'nowrap', flexShrink: 0, marginTop: 2,
        }}>
          No citation
        </span>
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ fontSize: 15, color: C.text, lineHeight: 1.6, marginBottom: 6 }}>{claim.claim}</div>
          {claim.note && <div style={{ fontSize: 13, color: C.muted, lineHeight: 1.55 }}>{claim.note}</div>}
        </div>
      </div>
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

function Footer() {
  return (
    <footer style={{ borderTop: `1px solid ${C.border}`, padding: '32px 24px 44px', textAlign: 'center' }}>
      <div style={{ fontSize: 12, color: C.muted, lineHeight: 1.75, maxWidth: 640, margin: '0 auto 16px' }}>
        Everything runs in your own browser. Your paper, sources, and API key are never sent to or stored on
        any server we run — we don't operate one. Your key goes straight from your browser to the AI provider
        you choose, and we never see or keep a copy of it. You're responsible for that provider's own usage,
        costs, and terms.
      </div>
      <div style={{ fontSize: 11.5, color: C.faint, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8 }}>
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

function SettingsModal({ provider, baseUrl, model, apiKey, huntContradictions, providerInfo,
                          onProvider, onBaseUrl, onModel, onApiKey, onToggleHunt, onClose }) {
  return (
    <div onClick={onClose} style={{
      position: 'fixed', inset: 0, background: 'rgba(4,7,14,0.72)', backdropFilter: 'blur(2px)',
      display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 50, padding: 20,
    }}>
      <div onClick={e => e.stopPropagation()} style={{
        background: C.panel, border: `1px solid ${C.border}`, borderRadius: 16,
        padding: '32px 34px', width: '100%', maxWidth: 440, boxShadow: '0 24px 70px rgba(0,0,0,0.55)',
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
            <select value={provider} onChange={e => onProvider(e.target.value)} style={inputStyle}>
              {Object.entries(PROVIDER_DEFAULTS).map(([key, d]) => <option key={key} value={key}>{d.label}</option>)}
            </select>
          </div>
          {provider !== 'anthropic' && (
            <div>
              <div style={fieldLabelStyle}>Base URL</div>
              <input value={baseUrl} onChange={e => onBaseUrl(e.target.value)} placeholder="API base URL" style={inputStyle} />
            </div>
          )}
          <div>
            <div style={fieldLabelStyle}>Model</div>
            <input value={model} onChange={e => onModel(e.target.value)} placeholder="Model name" style={inputStyle} />
          </div>
          <div>
            <div style={fieldLabelStyle}>API key</div>
            <input type="password" value={apiKey} onChange={e => onApiKey(e.target.value)}
                   placeholder={providerInfo.keyRequired ? 'API key' : 'Optional for local servers'} style={inputStyle} />
          </div>
          <label style={{ display: 'flex', alignItems: 'center', gap: 9, fontSize: 13, color: C.muted, cursor: 'pointer' }}>
            <input type="checkbox" checked={huntContradictions} onChange={onToggleHunt} />
            Hunt for contradictions (separate pass, ~2× calls)
          </label>
          <div style={{ fontSize: 11.5, color: C.faint, lineHeight: 1.65 }}>
            Stored only in this browser (localStorage), sent only to the base URL above.
            {!providerInfo.supportsPdf && ' This provider reads text only — PDFs need Anthropic.'}
          </div>
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
  const [provider, setProvider] = useState(() => localStorage.getItem('sv_provider') || 'anthropic');
  const [apiKey,   setApiKey]   = useState(() => localStorage.getItem('sv_api_key') || '');
  const [baseUrl,  setBaseUrl]  = useState(() => localStorage.getItem('sv_base_url') || PROVIDER_DEFAULTS.anthropic.baseUrl);
  const [model,    setModel]    = useState(() => localStorage.getItem('sv_model') || PROVIDER_DEFAULTS.anthropic.model);
  const [showSettings, setShowSettings] = useState(false);
  const [huntContradictions, setHuntContradictions] = useState(() => localStorage.getItem('sv_hunt') !== '0');

  const [paper,    setPaper]    = useState(null);
  const [paperTxt, setPaperTxt] = useState('');
  const [refs,     setRefs]     = useState([]);
  const [result,   setResult]   = useState(null);
  const [loading,  setLoading]  = useState(false);
  const [runStatus, setRunStatus] = useState('');
  const [liveChunk, setLiveChunk] = useState(null); // { current, total } for the active run only
  const [err,      setErr]      = useState(null);
  const [copied,   setCopied]   = useState(false);
  const [progress, setProgress] = useState(null); // paused/resumable snapshot, persisted
  const progressUpload = useRef();

  const providerInfo = PROVIDER_DEFAULTS[provider];

  useEffect(() => {
    try { const raw = localStorage.getItem('sv_progress'); if (raw) setProgress(JSON.parse(raw)); } catch {}
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

  const handlePaper = async file => {
    try { const d = await readFile(file); setPaper(d); setPaperTxt(d.content || ''); }
    catch (e) { setErr('Could not read paper: ' + e.message); }
  };
  const handleRefs = async files => {
    try { const loaded = await Promise.all(files.map(readFile)); setRefs(prev => [...prev, ...loaded]); }
    catch (e) { setErr('Could not read reference: ' + e.message); }
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
    a.href = url; a.download = `loupe-progress-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };
  const uploadProgress = async file => {
    try {
      const snap = JSON.parse(await file.text());
      saveProgress(snap);
      setPaper(snap.paper || null);
      setPaperTxt(snap.paper?.content || '');
      setRefs(snap.refs || []);
      setErr(null);
    } catch (e) { setErr('Could not read progress file: ' + e.message); }
  };

  // Shared by both a fresh run and a resume — resuming must not skip these.
  const validate = ({ forResume } = {}) => {
    const hasPdf = paper?.type === 'pdf' || refs.some(r => r.type === 'pdf');
    if (providerInfo.keyRequired && !apiKey.trim()) return `Enter your ${providerInfo.label} API key.`;
    if (!baseUrl.trim()) return 'Enter the API base URL.';
    if (!model.trim()) return 'Enter a model name.';
    if (hasPdf && !providerInfo.supportsPdf) return 'PDF files need the Anthropic provider — switch provider, or replace the PDF with .txt/.docx.';
    if (!forResume) {
      const text = paperTxt.trim() || paper?.content || '';
      const hasPaper = !!text || paper?.type === 'pdf';
      if (!hasPaper) return 'No paper text to verify.';
      if (!refs.length) return 'Upload at least one reference source.';
    }
    return null;
  };

  const runChunked = async resumeFrom => {
    const text = paperTxt.trim() || paper?.content || '';
    const chunks = resumeFrom?.chunks || chunkText(text);
    let cited = resumeFrom?.citedClaims || [];
    let uncited = resumeFrom?.uncitedClaims || [];
    let contradictions = resumeFrom?.contradictions || [];
    let i = resumeFrom?.chunkIndex || 0;

    const refParts = buildRefParts(refs, providerInfo.kind);
    setLoading(true); setErr(null); setResult(null);
    setLiveChunk({ current: i, total: chunks.length });

    for (; i < chunks.length; i++) {
      setRunStatus(`Checking chunk ${i + 1} of ${chunks.length}…`);
      const paperParts = [{ type: 'text', text: `[PAPER EXCERPT ${i + 1}/${chunks.length}]\n${chunks[i]}` }];
      try {
        const verdict = await callLLM({
          kind: providerInfo.kind, apiKey: apiKey.trim(), baseUrl: baseUrl.trim(), model: model.trim(),
          parts: [...refParts, ...paperParts], instructions: claimInstructions({ whole: false }),
        });
        cited = cited.concat(verdict.citedClaims || []);
        uncited = uncited.concat(verdict.uncitedClaims || []);

        if (huntContradictions) {
          setRunStatus(`Checking chunk ${i + 1} of ${chunks.length} for contradictions…`);
          const cv = await callLLM({
            kind: providerInfo.kind, apiKey: apiKey.trim(), baseUrl: baseUrl.trim(), model: model.trim(),
            parts: [...refParts, ...paperParts], instructions: CONTRADICTION_INSTRUCTIONS,
          });
          contradictions = contradictions.concat(cv.contradictions || []);
        }
      } catch (e) {
        const snap = { chunks, chunkIndex: i, citedClaims: cited, uncitedClaims: uncited, contradictions, paper, refs };
        saveProgress(snap);
        setLoading(false);
        setLiveChunk(null);
        setErr(`Stopped at chunk ${i + 1} of ${chunks.length}: ${e.message}. Progress is saved — click Resume to continue, or download it below.`);
        return;
      }
      setLiveChunk({ current: i + 1, total: chunks.length });
      saveProgress({ chunks, chunkIndex: i + 1, citedClaims: cited, uncitedClaims: uncited, contradictions, paper, refs });
    }

    setResult(buildFinalResult(cited, uncited, contradictions));
    clearProgress();
    setLoading(false);
    setLiveChunk(null);
    setRunStatus('');
  };

  const runWholeDocument = async () => {
    setLoading(true); setErr(null); setResult(null);
    setRunStatus('Reading sources and checking claims…');
    const refParts = buildRefParts(refs, providerInfo.kind);
    const paperParts = [filePart(paper, 'PAPER'), { type: 'text', text: '[The document above is the PAPER TO VERIFY]' }];
    try {
      const verdict = await callLLM({
        kind: providerInfo.kind, apiKey: apiKey.trim(), baseUrl: baseUrl.trim(), model: model.trim(),
        parts: [...refParts, ...paperParts], instructions: claimInstructions({ whole: true }),
      });
      let contradictions = [];
      if (huntContradictions) {
        setRunStatus('Hunting for contradictions…');
        const cv = await callLLM({
          kind: providerInfo.kind, apiKey: apiKey.trim(), baseUrl: baseUrl.trim(), model: model.trim(),
          parts: [...refParts, ...paperParts], instructions: CONTRADICTION_INSTRUCTIONS,
        });
        contradictions = cv.contradictions || [];
      }
      setResult(buildFinalResult(verdict.citedClaims || [], verdict.uncitedClaims || [], contradictions));
    } catch (e) { setErr('Verification failed: ' + e.message); }
    finally { setLoading(false); setRunStatus(''); }
  };

  const run = () => {
    const problem = validate();
    if (problem) { setErr(problem); return; }
    if (paper?.type === 'pdf') runWholeDocument();
    else runChunked();
  };

  const resume = () => {
    if (!progress) return;
    const problem = validate({ forResume: true });
    if (problem) { setErr(problem); return; }
    runChunked(progress);
  };

  const copyReport = () => {
    if (!result) return;
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
    navigator.clipboard.writeText(lines.join('\n'));
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const paperText = paperTxt.trim() || paper?.content || '';
  const hasPaper  = !!paperText || paper?.type === 'pdf';
  const canRun    = (!providerInfo.keyRequired || !!apiKey.trim()) && hasPaper && refs.length > 0 && !loading;
  const isChunkable = paper?.type !== 'pdf' && paperText;
  const estChunks = isChunkable ? chunkText(paperText).length : 1;
  const estCalls  = estChunks * (huntContradictions ? 2 : 1);
  const isPaused  = progress && progress.chunkIndex < progress.chunks?.length;

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
            <button onClick={() => setResult(null)} style={ghostBtnStyle}>
              <ArrowLeft size={13} /> New verification
            </button>
          )}
          {result && (
            <button onClick={copyReport} style={ghostBtnStyle}>
              <Copy size={13} /> {copied ? 'Copied!' : 'Copy report'}
            </button>
          )}
          <button onClick={() => setShowSettings(true)} style={ghostBtnStyle}>
            <Settings size={14} /> {providerInfo.label}
          </button>
        </div>
      </div>

      {showSettings && (
        <SettingsModal
          provider={provider} baseUrl={baseUrl} model={model} apiKey={apiKey}
          huntContradictions={huntContradictions} providerInfo={providerInfo}
          onProvider={handleProvider} onBaseUrl={handleBaseUrl} onModel={handleModel}
          onApiKey={handleApiKey} onToggleHunt={toggleHunt}
          onClose={() => setShowSettings(false)}
        />
      )}

      {/* Main content */}
      <div className="main-area">

        {!result && !loading && (
          <div className="setup">
            <div style={{ textAlign: 'center' }}>
              <h1 style={{ fontSize: 30, fontWeight: 800, letterSpacing: '-0.02em', margin: 0 }}>Verify a paper</h1>
              <p style={{ fontSize: 14.5, color: C.muted, marginTop: 10, lineHeight: 1.6, maxWidth: 560, marginLeft: 'auto', marginRight: 'auto' }}>
                Upload your paper and its reference sources. Every cited claim gets checked against them,
                and anything stated with no citation gets flagged for your own review.
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
                <div style={labelStyle}>Paper to verify <span style={{ color: C.red, fontWeight: 400, textTransform: 'none', letterSpacing: 0 }}>required</span></div>
                <div style={{ background: C.card, border: `1px solid ${paper ? C.teal + '44' : C.border}`, borderRadius: 12, overflow: 'hidden' }}>
                  {paper ? (
                    <div style={{ padding: '18px 20px', display: 'flex', alignItems: 'center', gap: 12 }}>
                      <FileText size={18} color={C.teal} />
                      <div style={{ flex: 1, overflow: 'hidden' }}>
                        <div style={{ fontSize: 13.5, fontWeight: 600, color: C.teal, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{paper.name}</div>
                        <div style={{ fontSize: 12, color: C.muted, marginTop: 3 }}>
                          {paper.type === 'pdf' ? 'PDF document · single-pass (no chunking)' : `${(paper.content || '').split(' ').filter(Boolean).length} words`}
                        </div>
                      </div>
                      <button onClick={() => { setPaper(null); setPaperTxt(''); }} style={{ background: 'none', border: 'none', color: C.red, cursor: 'pointer' }}>
                        <Trash2 size={15} />
                      </button>
                    </div>
                  ) : (
                    <DropZone label="Upload paper" sub=".txt · .docx · .pdf" onFile={handlePaper} flush />
                  )}
                  <div style={{ borderTop: `1px solid ${C.border}` }} />
                  <textarea value={paperTxt} onChange={e => setPaperTxt(e.target.value)} placeholder="Or paste the paper text here…"
                            style={{
                              width: '100%', minHeight: 130, background: 'transparent', border: 'none', outline: 'none',
                              padding: '16px 20px', color: C.text, fontSize: 13, lineHeight: 1.65, resize: 'vertical', fontFamily: 'inherit',
                            }} />
                </div>
                {isChunkable && (
                  <div style={{ fontSize: 11.5, color: C.faint, marginTop: 8 }}>
                    ~{estChunks} chunk{estChunks !== 1 ? 's' : ''} · ~{estCalls} API call{estCalls !== 1 ? 's' : ''} to check every cited claim
                  </div>
                )}
              </div>

              <div>
                <div style={labelStyle}>Reference sources</div>
                <div style={{ background: C.card, border: `1px solid ${C.border}`, borderRadius: 12, overflow: 'hidden' }}>
                  <DropZone label="Upload sources (multiple)" sub=".txt · .docx · .pdf" multi onFiles={handleRefs} flush />
                </div>
                {refs.length > 0 ? (
                  <div style={{ marginTop: 14, display: 'flex', flexWrap: 'wrap', gap: 8 }}>
                    {refs.map((r, i) => <FileChip key={i} name={r.name} onRemove={() => setRefs(prev => prev.filter((_, j) => j !== i))} />)}
                  </div>
                ) : (
                  <div style={{ fontSize: 12.5, color: C.faint, marginTop: 14, lineHeight: 1.6 }}>
                    The documents you want claims cross-checked against — papers, reports, datasets, anything with the facts your paper cites.
                  </div>
                )}
              </div>
            </div>

            <div style={{ textAlign: 'center' }}>
              <button onClick={run} disabled={!canRun} style={{
                background: canRun ? C.teal : C.card, color: canRun ? '#070C18' : C.faint,
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
                    background: 'none', border: 'none', color: C.faint, cursor: 'pointer',
                    fontSize: 12, textDecoration: 'underline', textUnderlineOffset: 3,
                  }}>
                    Resuming on a different machine? Load a progress file
                  </button>
                </div>
              )}
            </div>

            <p style={{ fontSize: 12.5, color: C.faint, lineHeight: 1.7, textAlign: 'center', maxWidth: 620, margin: '0 auto' }}>
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

            <div style={{ fontSize: 12, fontWeight: 700, color: C.muted, textTransform: 'uppercase', letterSpacing: '0.08em', marginTop: result.contradictions.length ? 36 : 0, marginBottom: 16 }}>
              Cited Claims — Checked Against Sources
            </div>
            {result.citedClaims.map((c, i) => <ClaimCard key={i} claim={c} />)}

            {result.uncitedClaims.length > 0 && (
              <>
                <div style={{ fontSize: 12, fontWeight: 700, color: C.muted, textTransform: 'uppercase', letterSpacing: '0.08em', marginTop: 36, marginBottom: 10 }}>
                  Claims With No Citation
                </div>
                <div style={{ fontSize: 13, color: C.muted, lineHeight: 1.6, marginBottom: 16 }}>
                  Not checked against your sources — review these yourself to confirm they're your own analysis.
                </div>
                {result.uncitedClaims.map((c, i) => <UncitedClaimCard key={i} claim={c} />)}
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

        .main-area { padding: 56px 32px 80px; }
        .setup { max-width: 920px; margin: 0 auto; display: flex; flex-direction: column; gap: 40px; }
        .setup-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 32px; }
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
