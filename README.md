# Loupe — AI Research Paper Citation & Reference Verifier

**Loupe** is a free, open-source tool that checks whether a research paper's citations actually say what the paper claims they say. Upload a paper and its reference sources, and Loupe cross-references every cited claim against them, flags claims made with no citation at all, and runs a dedicated pass to hunt for outright contradictions between the paper and its own sources — entirely in your browser, using your own AI provider API key.

No account, no upload to a third-party server, no internet search. It's a citation-accuracy checker and reference-verification tool for anyone who needs to sanity-check a paper, literature review, thesis chapter, or AI-assisted draft before it goes out the door: authors, reviewers, editors, and students.

> If you searched "Loupe" and landed here expecting a jeweler's magnifying glass — that's exactly where the name comes from. This is that same idea of close, careful examination, applied to a paper's citations.

## What it does

Loupe reads through a paper and splits its claims into three things worth knowing:

- **Cited claims** — anything with an in-text citation gets checked against your uploaded sources, with a verdict (supported / partial / unsupported / contradicted), a quoted excerpt as evidence, and a confidence level.
- **Claims with no citation** — factual-sounding statements with nothing attached, surfaced separately so you can confirm they're your own analysis rather than something that quietly needed a source.
- **Contradictions** — a dedicated adversarial pass that specifically hunts for places where a source says the opposite of what the paper claims, since that's a different search than "does this support the claim?" and gets missed if you only ask one question.

Long papers are split into chunks so every cited claim gets checked — not just a top handful — and a run that gets interrupted (rate limit, network drop, ran out of tokens) saves its progress automatically, so you can resume instead of starting over.

## Privacy — read this first

Loupe has no backend and no account system. Everything — reading your paper, reading your sources, calling the AI provider, showing you the results — happens in your own browser tab.

- **We do not keep your API key.** It's stored only in your own browser's `localStorage`, on your own device. It is never sent to us, because there is no "us" to send it to — Loupe doesn't run a server at all.
- **Your paper and sources never leave your machine except to the AI provider you pick.** Whatever provider you choose (Anthropic, OpenAI, Google, Groq, OpenRouter, Hugging Face, or your own local model) is the only place your text or your key is ever sent, directly from your browser.
- **You are responsible for that provider.** Loupe doesn't manage billing, rate limits, or data-retention policy for OpenAI, Anthropic, or anyone else — that's between you and the provider whose key you paste in. Check their terms and pricing before you run a large paper through them.

## What it does *not* do

- It does not search the internet. It only checks your paper against the specific source files you upload.
- It does not store, log, or transmit anything to a server we control — there isn't one.
- It does not tell you a paper is "true." It tells you whether a specific claim is backed by a specific passage in a specific file you gave it, and flags what isn't.

## Providers

The settings panel presents these in three tiers, in order:

1. **Local — no API key, no setup** *(recommended, default)* — runs entirely in your browser, nothing to install, nothing to pay for, see below.
2. **Local AI (Ollama, LM Studio, vLLM, …)** — any OpenAI-compatible server running on your own machine. LLM-quality reasoning, still no API key, but you need that server running.
3. **Bring your own API key**, tucked behind a toggle since it's a bigger ask than one click. Instead of a dropdown, there's one text box: type a name (`claude`, `deepseek`, `openrouter`, …) and matching providers show up as clickable suggestions — click one and the base URL, model, and everything else fill in for you. Recognized out of the box:

   | Provider | Notes |
   |---|---|
   | Anthropic (Claude) | Reads PDF bytes natively |
   | OpenAI | |
   | Google (Gemini) | via Google's OpenAI-compatible endpoint — has a free tier |
   | Groq | has a free tier |
   | OpenRouter | |
   | Hugging Face | via the Hugging Face Inference Router |
   | DeepSeek | |

   Type something we don't recognize and it just becomes a custom entry — fill in the base URL, model, and key yourself. If someone has an API key for a provider we don't know by name, they almost certainly know its base URL already.

All of these except Anthropic speak the same OpenAI-compatible `chat/completions` shape, so adding another one is just adding a base URL — that's also why an unrecognized custom entry defaults to that shape rather than asking which format it speaks.

**PDF support**: every provider handles PDFs now, just not identically. Anthropic reads the PDF's bytes natively (best fidelity — tables, figures, layout). Every other provider — including the local-AI and bring-your-own-key ones — gets the text [`pdfjs-dist`](https://github.com/mozilla/pdf.js) extracts from the PDF client-side, same as a `.txt` upload from that point on. That's a deliberate "support it as broadly as possible without pretending it's the same fidelity everywhere" choice: a PDF with complex tables may extract messily outside Anthropic, but it isn't blocked.

### Local — no API key, no setup

This is the recommended default: no account, no payment, nothing to configure. It doesn't call any LLM. Instead:

1. **Retrieval**: reference documents are split into sentences and indexed with [BM25](https://en.wikipedia.org/wiki/Okapi_BM25) (classic lexical search), then an embedding model ([`all-MiniLM-L6-v2`](https://huggingface.co/Xenova/all-MiniLM-L6-v2) by default) reranks the top candidates semantically.
2. **Claim detection**: sentences in your paper are flagged as claims by simple rules — carries a citation (APA/Chicago author-date, or bracketed/numeric like `[3]`), contains a number/statistic, uses a causal verb ("shows", "demonstrates", "found that"), or a comparative ("higher than", "increased").
3. **Reasoning**: for each cited claim, an NLI (natural-language-inference) model ([`nli-deberta-v3-base`](https://huggingface.co/Xenova/nli-deberta-v3-base) by default) checks whether each of the top-5 retrieved reference sentences actually supports, contradicts, or is unrelated to the claim.

Both models (~22MB + ~233MB by default) download once from the Hugging Face Hub the first time you use this option and are cached by the browser — after that, verification runs fully offline. Everything happens inside a Web Worker so the page stays responsive during inference. Model choice, and the (advanced, collapsed by default) retrieval-method setting, live in the settings panel.

#### Model quality — measured, not guessed

`bench/run.mjs` runs a small hand-built test set against every model option using the exact same classification code the app runs in production (`src/lib/nli.js`), and prints real accuracy numbers. Run it yourself with `npm run bench`. Current results — 6 retrieval test cases, 14 reasoning test cases, so treat this as real signal from a small sample, not a rigorous benchmark:

| Embedding model | Retrieval accuracy |
|---|---|
| `all-MiniLM-L6-v2` (default) | 100% — smallest and fastest of the tied options |
| `e5-small-v2` | 100% |
| `all-MiniLM-L12-v2` | 100% |
| `snowflake-arctic-embed-s` | 100% |

`bge-small-en-v1.5` was tested and dropped: 83%, worse than the options above while also being larger than the default — no axis it won on. `bge-base-en-v1.5` tied at 100% but was later removed too: on a real paper it showed no accuracy edge over the options above despite being ~4x their size and per-embedding time.

| NLI model | Reasoning accuracy |
|---|---|
| `nli-deberta-v3-base` (default) | 79%, ~350MB |
| `deberta-v3-base-zeroshot-v2.0` | 86% — the most accurate option, but ~740MB. Not the default: its source repo only publishes an unquantized ONNX export (no `model_quantized.onnx` like the other models here), so it needs `dtype: 'fp32'` forced explicitly in `NLI_MODELS` or the browser runtime 404s trying to fetch a quantized file that doesn't exist — and even fixed, it's over twice the download for 7 more points, which isn't the right trade-off as a default |
| `mobilebert-uncased-mnli` | 50%, ~100MB — but missed *every* contradiction case in testing (called them "unrelated"); fine for a quick supported/unsupported read, not for the contradiction-hunting feature |

`nli-deberta-v3-small` (36%) was also tested and dropped — performed close to chance on anything beyond a direct match. Whenever a new model option is added, re-run the benchmark **and check which ONNX variants actually exist in that repo** before wiring it in — never hand-write a percentage or assume a quantized file exists.

**Be honest about the trade-off**: even with the best-measured models, this is meaningfully more limited than an LLM-based provider. Claim detection is rule-based, not semantic understanding, so it will miss claims an LLM would catch and occasionally flag ones that aren't real claims. NLI models are also known to be less reliable than LLMs at numeric reasoning (e.g. distinguishing "a 40% reduction" from "a 45% reduction") and multi-sentence context. The larger default models and wider evidence pool (top-5, not top-3) push it as close to LLM-level judgment as this approach reasonably gets — but it's a different technique, not a compressed copy of an LLM, and it won't always agree with what an LLM-based provider would say about the same claim.

## Running it

Requires [Node.js](https://nodejs.org) 18+.

```bash
git clone https://github.com/YasirM0/loupe.git
cd loupe
npm install
npm run dev
```

Open the URL it prints, click the settings icon to add an API key for whichever provider you're using, upload your paper and reference sources, and click **Verify Claims**.

### Desktop app

[`src-tauri/`](src-tauri) wraps this into a [Tauri](https://tauri.app) desktop app — a real installer (`.exe`/`.msi` on Windows, `.dmg` on macOS, `.AppImage`/`.deb` on Linux), no terminal required to run it once installed.

Pushing a version tag (`git tag v0.1.0 && git push --tags`) triggers [`.github/workflows/release.yml`](.github/workflows/release.yml), which builds all three natively on their own OS in CI and attaches the installers to a [GitHub Release](https://github.com/YasirM0/loupe/releases) as a draft — cross-compiling desktop installers locally is unreliable, so this repo doesn't try to.

To build one yourself locally instead:

```bash
npm install
npm run tauri build
```

This requires the [Tauri prerequisites](https://v2.tauri.app/start/prerequisites/) for your OS (Rust, plus WebView2 on Windows / webkit2gtk on Linux — macOS needs nothing extra).

## How verification works

1. **Chunking.** Text/docx papers are split into ~900-word chunks along paragraph boundaries. Each chunk is checked exhaustively — every cited claim in it, not a top-N sample — which is what actually fixes coverage on long papers, rather than asking a bigger model to somehow do better. PDF papers are checked in a single pass (up to 20 claims), since we can't split a PDF into text chunks without a separate parser.
2. **Grounding rules.** Every verdict is produced under a strict system prompt: no verdict without a literal quoted passage, no use of the model's own background knowledge to fill gaps, `CONTRADICTED` only from explicit contradicting text, and a citation's mere presence in the paper is never treated as evidence on its own — the underlying content still has to actually appear in your uploaded sources.
3. **Contradiction pass.** A second, separately-framed call asks only "what here contradicts the paper?" — deliberately not mixed into the support-checking call, since a single prompt biases toward the framing it's given and under-searches for the other thing.
4. **Resumability.** Progress is saved to `localStorage` after every chunk. If a run stops partway (you'll see exactly which chunk it stopped at and why), click **Resume** to continue, or **Download** the progress file to move it to another browser/machine and pick up there via **Load progress file**.

## Works with small and local models, not just frontier ones

Loupe isn't built assuming you're paying for a large hosted model. Smaller and local models (a 3B–7B model through Ollama, LM Studio, etc.) are far more prone to two specific failures under a long, rule-heavy prompt: dropping the connection under memory pressure, and not strictly following a "respond with only JSON" instruction. Rather than surfacing those as a crash, Loupe retries automatically (up to 3 attempts, with a short backoff) on both connection failures and malformed JSON, and pulls the JSON object out of a response even if the model added commentary around it. This trades a bit of time for reliability on purpose — the response quality target doesn't move depending on which model you bring; the model choice is entirely up to you.

## Cost note

Chunking means more API calls than a single-shot summary would — each chunk resends your source documents so accurate checking can happen against the full context. On Anthropic, the source documents are marked for prompt caching to keep that from multiplying token cost by the chunk count; other providers don't get that optimization here yet. The app shows an estimated chunk/call count before you run. Automatic retries (above) can add a couple more calls on top of that estimate when a response needs a second try.

## Frequently asked questions

**Is Loupe free?** Loupe itself is free and open source (MIT). You pay only for whatever AI provider usage you generate with your own key.

**Does Loupe store my paper?** No. There's no server, so there's nothing to store it in. Close the tab and it's gone unless you explicitly downloaded a progress file yourself.

**Can I use it without an API key?** You need a key (or a running local model) for whichever provider you pick — Loupe itself doesn't include or proxy any AI access.

**Does it work offline?** The app loads and reads your files offline, but checking claims requires reaching whichever AI provider you've configured — including a local model server on your own machine, which counts as "offline" from the internet's point of view.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## Author

Built by [Yasir Mohammed](https://github.com/YasirM0).

## License

MIT — see [LICENSE](LICENSE).
