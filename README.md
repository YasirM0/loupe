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

Pick any of these, or point it at a local model server:

| Provider | Notes |
|---|---|
| Anthropic (Claude) | Only provider with native PDF support — Claude reads PDF bytes directly |
| OpenAI | |
| Google (Gemini) | via Google's OpenAI-compatible endpoint |
| Groq | |
| OpenRouter | |
| Hugging Face | via the Hugging Face Inference Router |
| Local / custom | any OpenAI-compatible server — Ollama, LM Studio, vLLM, etc. |

All of these except Anthropic speak the same OpenAI-compatible `chat/completions` shape, so adding another one is just adding a base URL. PDF papers and PDF sources currently require Anthropic, since Claude parses PDF bytes natively — everyone else needs `.txt` or `.docx`, since generic chat-completions endpoints don't take raw documents.

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

## Cost note

Chunking means more API calls than a single-shot summary would — each chunk resends your source documents so accurate checking can happen against the full context. On Anthropic, the source documents are marked for prompt caching to keep that from multiplying token cost by the chunk count; other providers don't get that optimization here yet. The app shows an estimated chunk/call count before you run.

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
