// UI string dictionary — every user-facing string in Loupe.jsx, keyed so
// `t.someKey` (or `t.someKey(args)` for templated ones) replaces the
// literal. English (`en`) is the original, unmodified copy — nothing here
// changes what an English-language user sees. `id` is added below,
// verified against the app's actual local pipeline (see bench/run.mjs's
// 2026-08-30 Indonesian comparison: 100% retrieval / 86% NLI, matching the
// English baseline exactly, once claim/evidence text is translated to
// English before it reaches the embedding/NLI models — see
// dictionary.id-en.js and inference.worker.js).
//
// Functions (not plain strings) are used wherever word order, pluralization,
// or count-dependent phrasing differs between languages — concatenating
// translated fragments around a shared template breaks exactly the cases
// that matter most (Indonesian has no plural marking at all; English's
// "claim"/"claims" split doesn't map onto that cleanly).

const en = {
  // ── Header ──
  tagline: 'Checks every citation, hunts for contradictions · no internet search',
  newVerification: 'New verification',
  copyReport: 'Copy report',
  copied: 'Copied!',
  downloadReport: 'Download report',
  checkingWith: 'Checking with: ',
  settingsTooltip: 'Change what checks your claims — model, provider, or API key',
  languageTooltip: 'Interface and check language',

  // ── Setup screen ──
  setupTitle: "Verify your paper's claims",
  setupSubtitle: 'Checks every citation against your sources and flags anything stated without one.',
  runsInBrowserBadge: 'Runs entirely in your browser — nothing is stored on any server',
  pausedAt: (chunkIndex, total) => `Paused at chunk ${chunkIndex} of ${total} from a previous run`,
  resume: 'Resume',
  download: 'Download',
  paperToVerify: 'Paper to verify',
  required: 'required',
  restorePreviousPaper: 'Restore previous paper',
  uploadPaper: 'Upload paper',
  fileTypesHint: '.txt · .docx · .pdf',
  dragAndDrop: 'or drag and drop',
  orPasteTextInstead: 'Or paste text instead',
  pasteTextPlaceholder: 'Paste the paper text here…',
  clearText: 'Clear text',
  backToFileUpload: '← Back to file upload',
  pdfSinglePass: 'PDF document · single-pass (native, no chunking)',
  pdfExtractedText: 'PDF document · read as extracted text',
  wordCount: n => `${n} word${n !== 1 ? 's' : ''}`,
  chunkEstimate: (chunks, calls) => `~${chunks} chunk${chunks !== 1 ? 's' : ''} · ~${calls} API call${calls !== 1 ? 's' : ''} to check every cited claim`,
  browserClaimEstimate: n => `~${n} cited claim${n !== 1 ? 's' : ''} detected — no API calls, runs entirely on-device`,
  referenceSources: 'Reference sources',
  restorePreviousSources: 'Restore previous sources',
  uploadSources: 'Upload sources (multiple)',
  addMoreSources: 'Add more sources',
  addAsManyAsNeeded: 'Add as many as you need',
  refsHelperText: 'The documents you want claims cross-checked against — papers, reports, datasets, anything with the facts your paper cites.',
  verifyClaims: 'Verify Claims',
  resumeOnDifferentMachine: 'Resuming on a different machine? Load a progress file',
  setupFooterNote: 'Text/docx papers are split into chunks so every cited claim gets checked, not just the top few — PDF papers run in a single pass. A dedicated pass hunts for contradictions. If a run is interrupted, progress saves automatically so you can resume it.',
  readingFiles: multi => `Reading file${multi ? 's' : ''}…`,

  // ── Loading screen ──
  defaultLoadingStatus: 'Reading sources and checking claims…',
  oneTimeDownloadNote: 'One-time download — needs internet now, then this runs offline every time after.',
  embeddingModelLabel: 'Embedding model',
  nliModelLabel: 'NLI model',
  translationModelLabel: 'Translation model',

  // ── Results — score header ──
  citationAccuracyScore: 'Citation Accuracy Score (adjusted)',
  rawScoreLabel: n => `Raw score: ${n}/100`,
  countSupported: n => `${n} supported`,
  countPartial: n => `${n} partial`,
  countUnsupported: n => `${n} unsupported`,
  countContradicted: n => `${n} contradicted`,
  countUncited: n => `${n} uncited`,

  // ── Results — source coverage ──
  sourceCoverage: 'Source Coverage',
  coverageFound: (found, total) => `${found} of ${total} cited sources found in corpus`,
  referencesExcludedNote: n => `${n} reference entries identified and excluded from verification.`,
  missingSourcesIntro: 'The following cited sources were not found in the verification corpus:',
  uploadBeforeRerun: 'Upload these before re-running for a reliable score.',
  extractionWarningsIntro: 'Sources with extraction warnings:',
  extractionWarningBody: name => `${name} may have limited text extractability (slide-deck or scanned format). Claims citing this source may be marked UNSUPPORTED due to extraction limitations rather than genuine absence from the source.`,

  // ── Results — claims ──
  contradictionsFound: 'Contradictions Found',
  allClaims: 'All Claims',
  uncited: 'Uncited',
  citedClaimsHeading: 'Cited Claims — Checked Against Sources',
  uncitedClaimsHeading: 'Claims With No Citation',
  uncitedClaimsSubtitle: "Not checked against your sources — review these yourself to confirm they're your own analysis.",
  noCitation: 'No citation',
  markedAsYours: 'Marked as yours',
  thisIsMine: 'This is mine',
  helpMeCiteThis: 'Help me cite this',
  closestMatch: 'Closest match in your sources — verify it yourself before citing',
  sourcesChecked: 'Sources checked',

  // status/category labels
  statusSupported: 'Supported',
  statusPartial: 'Partial',
  statusUnsupported: 'No source',
  statusContradicted: 'Contradicted',
  unsupportedSourceNotUploaded: 'SOURCE NOT UPLOADED',
  unsupportedExtractionLimited: 'EXTRACTION LIMITED',
  unsupportedNotFoundInSource: 'NOT FOUND IN SOURCE',
  categoryStatistic: 'Statistics without citation',
  categoryComparative: 'Comparative claims without citation',
  categoryInterpretive: 'Interpretive claims flagged by rule',

  // ── Settings modal ──
  connection: 'Connection',
  provider: 'Provider',
  recommendedBadge: 'Recommended',
  browserProviderLabel: 'Local — no API key, no setup',
  browserProviderDesc: 'Runs fully in your browser. Your paper, sources, and results never leave your device — nothing is uploaded anywhere, no account needed, and it keeps working offline after the first load.',
  localProviderLabel: 'Local AI (Ollama, LM Studio…)',
  localProviderDesc: 'Point at a model server running on your own machine (e.g. Ollama) for LLM-quality reasoning without an API key — needs that server running and a bit of setup.',
  bringYourOwnKey: 'Or bring your own API key (Claude, DeepSeek, OpenRouter…)',
  hide: 'Hide',
  providerName: 'Provider name',
  providerNamePlaceholder: 'e.g. claude, deepseek, openrouter…',
  notRecognizedNote: "Not one we recognize — that's fine, click away and fill in the base URL and model yourself below.",
  baseUrl: 'Base URL',
  baseUrlPlaceholder: 'e.g. https://api.example.com/v1',
  baseUrlLocalPlaceholder: 'e.g. http://localhost:11434/v1',
  model: 'Model',
  modelPlaceholder: 'Model name',
  apiKey: 'API key',
  apiKeyPlaceholder: 'API key',
  apiKeyOptionalPlaceholder: 'Optional for local servers',
  huntContradictionsLabel: 'Hunt for contradictions (separate pass, ~2× calls)',
  storedOnlyNote: 'Stored only in this browser (localStorage), sent only to the base URL above.',
  browserPipelineExplainer: 'No API key, no account, no server. Your first verification needs internet once, to download two small models to this browser (they\'re cached after that) — every verification after that runs fully offline, unless you switch to an API-key provider below, which always needs a connection. Quality is more limited than an LLM-based provider (rule-based claim detection, embedding + NLI reasoning instead of full language understanding), but there\'s nothing to pay for and nothing to configure. The defaults are already the best-measured options below (real test results, not a guess) — most people don\'t need to open "Advanced."',
  advancedModelSettings: 'Advanced model settings',
  embeddingModelRetrieval: 'Embedding model (retrieval)',
  nliModelReasoning: 'NLI model (reasoning)',
  retrievalMethod: 'Retrieval method',
  qualityOnTestSet: pct => `${pct}% on our test set`,
  done: 'Done',

  // ── Footer ──
  privacyParagraph: "Everything runs in your own browser. Your paper, sources, and API key are never sent to or stored on any server we run — we don't operate one. Your key goes straight from your browser to the AI provider you choose, and we never see or keep a copy of it. You're responsible for that provider's own usage, costs, and terms.",
  scilenePromoIntro: 'Once your citations check out, the next question is usually where to submit.',
  scilenePromoBody: '— a companion tool from the same author — matches your manuscript against journals indexed in Scopus, Web of Science, DOAJ, and SINTA, with a plain-language explanation for why each one fits. Same approach as here: no account, runs locally.',
  sourceOnGithub: 'Source on GitHub',
  builtBy: 'Built by',

  // ── Errors / validation ──
  enterApiKey: label => `Enter your ${label} API key.`,
  enterBaseUrl: 'Enter the API base URL.',
  enterModelName: 'Enter a model name.',
  noPaperText: 'No paper text to verify.',
  uploadAtLeastOneSource: 'Upload at least one reference source.',
  couldNotReadPaper: msg => `Could not read paper: ${msg}`,
  couldNotReadReference: msg => `Could not read reference: ${msg}`,
  couldNotReadProgressFile: msg => `Could not read progress file: ${msg}`,
  verificationFailed: msg => `Verification failed: ${msg}`,
  resumeWrongPipeline: 'This paused run is from an AI-provider verification, not the local browser pipeline — switch to the provider you were using (via the settings icon) and click Resume again.',
  stoppedAround: (batchLabel, total, msg) => `Stopped around ${batchLabel} of ${total}: ${msg}. Progress is saved — click Resume to continue, or download it below.`,

  // ── Run-status templates ──
  chunkLabel: i => `chunk ${i}`,
  chunksLabel: (i, j) => `chunks ${i}–${j}`,
  checkingBatch: (batchLabel, total) => `Checking ${batchLabel} of ${total}…`,
  checkingBatchRetrying: (batchLabel, total, attempt, max) => `Checking ${batchLabel} of ${total} — a response wasn't usable, retrying (${attempt}/${max})…`,
  responseRetrying: (attempt, max) => `A response wasn't usable, retrying (${attempt}/${max})…`,
  checkingAndHunting: 'Checking claims and hunting for contradictions…',
  loadingLocalModels: "Loading local models (first run downloads them, then they're cached)…",
  indexingSentences: (n, sources) => `Indexing ${n} sentences from ${sources} source${sources !== 1 ? 's' : ''}…`,
  indexingProgress: (done, total) => `Indexing sentence ${done} of ${total}…`,
  checkingCitedClaims: n => `Checking ${n} cited claim${n !== 1 ? 's' : ''}…`,
  checkingClaimProgress: (done, total) => `Checking claim ${done} of ${total}…`,

  // These two are worker-produced sentinel/display strings (inference.worker.js's
  // UNSUPPORTED-branch evidence text) — the worker itself stays language-
  // agnostic, so Loupe.jsx substitutes the translated version when building
  // each claim's display entry rather than threading `t` into the worker.
  notFoundInSources: 'not found in sources',
  noSimilarPassage: 'No sufficiently similar passage found in sources.',
  // Same substitution pattern as the two above — extractUncitedClaims in
  // textProcessing.js hardcodes this note in English since it has no
  // access to `t` (it's a pure text-processing function, not component-
  // aware); Loupe.jsx swaps it for the translated version when building
  // the uncited-claims list for the active language.
  flaggedByLocalRules: 'Flagged by local rules (number, comparison, or reasoning verb) — no citation attached.',
  liveChecked: 'Checked',
  liveSupported: 'Supported',
  livePartial: 'Partial',
  liveIssues: 'Issues',
  latestChecked: 'Latest checked',

  // ── Report summary (buildFinalResult/summarize) ──
  noCitedClaims: n => `No cited claims were found to check. ${n} sentence${n !== 1 ? 's' : ''} flagged as potentially uncited after filtering.`,
  reportSummary: ({ total, supported, partial, unsupported, sourceNotUploaded, extractionLimited, notFoundInSource, contradicted, uncitedCount }) =>
    `${total} cited claim${total !== 1 ? 's' : ''} checked. ${supported} supported, ${partial} partial, ${unsupported} unsupported `
    + `(${sourceNotUploaded} due to missing sources, ${extractionLimited} due to extraction limits, ${notFoundInSource} genuinely not found)`
    + `${contradicted ? `, ${contradicted} contradicted` : ''}. ${uncitedCount} sentence${uncitedCount !== 1 ? 's' : ''} flagged as potentially uncited after filtering.`,

  // ── Downloadable/copyable report text (buildReportText) ──
  reportTitle: 'LOUPE — SOURCE VERIFICATION REPORT',
  reportScoreLine: (raw, adjusted) => `Raw score: ${raw}/100   Adjusted score: ${adjusted}/100`,
  reportCoverageLine: (found, total) => `Source coverage: ${found} of ${total} cited sources found in corpus`,
  reportExtractionWarningsLine: names => `Sources with extraction warnings: ${names}`,
  reportReferencesLine: n => `References section: ${n} entries identified and excluded from verification.`,
  reportSourceCoverageHeading: 'SOURCE COVERAGE',
  reportMissingSourcesLine: names => `The following cited sources were not found in the verification corpus: ${names}.`,
  reportUploadBeforeRerun: 'Upload these before re-running for a reliable score.',
  reportContradictionsHeading: 'CONTRADICTIONS',
  reportCitedClaimsHeading: 'CITED CLAIMS',
  reportEvidenceLine: q => `Evidence: "${q}"`,
  reportSourceLine: name => `Source: ${name}`,
  reportUncitedHeading: 'CLAIMS WITHOUT A CITATION',
};

const id = {
  // ── Header ──
  tagline: 'Memeriksa setiap kutipan, mencari kontradiksi · tanpa pencarian internet',
  newVerification: 'Verifikasi baru',
  copyReport: 'Salin laporan',
  copied: 'Disalin!',
  downloadReport: 'Unduh laporan',
  checkingWith: 'Memeriksa dengan: ',
  settingsTooltip: 'Ubah apa yang memeriksa klaim Anda — model, penyedia, atau kunci API',
  languageTooltip: 'Bahasa antarmuka dan pemeriksaan',

  // ── Setup screen ──
  setupTitle: 'Verifikasi klaim makalah Anda',
  setupSubtitle: 'Memeriksa setiap kutipan terhadap sumber Anda dan menandai pernyataan apa pun yang tidak memiliki kutipan.',
  runsInBrowserBadge: 'Berjalan sepenuhnya di browser Anda — tidak ada yang disimpan di server mana pun',
  pausedAt: (chunkIndex, total) => `Dijeda di bagian ${chunkIndex} dari ${total} dari proses sebelumnya`,
  resume: 'Lanjutkan',
  download: 'Unduh',
  paperToVerify: 'Makalah untuk diverifikasi',
  required: 'wajib',
  restorePreviousPaper: 'Pulihkan makalah sebelumnya',
  uploadPaper: 'Unggah makalah',
  fileTypesHint: '.txt · .docx · .pdf',
  dragAndDrop: 'atau seret dan lepas',
  orPasteTextInstead: 'Atau tempel teks sebagai gantinya',
  pasteTextPlaceholder: 'Tempel teks makalah di sini…',
  clearText: 'Hapus teks',
  backToFileUpload: '← Kembali ke unggah file',
  pdfSinglePass: 'Dokumen PDF · satu tahap (native, tanpa pembagian)',
  pdfExtractedText: 'Dokumen PDF · dibaca sebagai teks yang diekstrak',
  wordCount: n => `${n} kata`,
  chunkEstimate: (chunks, calls) => `~${chunks} bagian · ~${calls} panggilan API untuk memeriksa setiap klaim yang dikutip`,
  browserClaimEstimate: n => `~${n} klaim terkutip terdeteksi — tanpa panggilan API, berjalan sepenuhnya di perangkat`,
  referenceSources: 'Sumber referensi',
  restorePreviousSources: 'Pulihkan sumber sebelumnya',
  uploadSources: 'Unggah sumber (bisa lebih dari satu)',
  addMoreSources: 'Tambah sumber lagi',
  addAsManyAsNeeded: 'Tambahkan sebanyak yang Anda perlukan',
  refsHelperText: 'Dokumen yang ingin Anda gunakan untuk memeriksa silang klaim — makalah, laporan, dataset, apa pun yang memuat fakta yang dikutip makalah Anda.',
  verifyClaims: 'Verifikasi Klaim',
  resumeOnDifferentMachine: 'Melanjutkan di perangkat lain? Muat file progres',
  setupFooterNote: 'Makalah teks/docx dibagi menjadi beberapa bagian agar setiap klaim yang dikutip diperiksa, bukan hanya beberapa teratas — makalah PDF diproses dalam satu tahap. Satu tahap khusus mencari kontradiksi. Jika proses terhenti, progres disimpan otomatis sehingga Anda dapat melanjutkannya.',
  readingFiles: () => 'Membaca file…',

  // ── Loading screen ──
  defaultLoadingStatus: 'Membaca sumber dan memeriksa klaim…',
  oneTimeDownloadNote: 'Unduhan satu kali — memerlukan internet sekarang, setelah itu berjalan offline setiap saat.',
  embeddingModelLabel: 'Model embedding',
  nliModelLabel: 'Model NLI',
  translationModelLabel: 'Model terjemahan',

  // ── Results — score header ──
  citationAccuracyScore: 'Skor Akurasi Kutipan (disesuaikan)',
  rawScoreLabel: n => `Skor mentah: ${n}/100`,
  countSupported: n => `${n} didukung`,
  countPartial: n => `${n} sebagian`,
  countUnsupported: n => `${n} tanpa sumber`,
  countContradicted: n => `${n} bertentangan`,
  countUncited: n => `${n} tanpa kutipan`,

  // ── Results — source coverage ──
  sourceCoverage: 'Cakupan Sumber',
  coverageFound: (found, total) => `${found} dari ${total} sumber yang dikutip ditemukan dalam korpus`,
  referencesExcludedNote: n => `${n} entri referensi teridentifikasi dan dikecualikan dari verifikasi.`,
  missingSourcesIntro: 'Sumber yang dikutip berikut tidak ditemukan dalam korpus verifikasi:',
  uploadBeforeRerun: 'Unggah sumber-sumber ini sebelum menjalankan ulang untuk mendapatkan skor yang andal.',
  extractionWarningsIntro: 'Sumber dengan peringatan ekstraksi:',
  extractionWarningBody: name => `${name} mungkin memiliki keterbatasan dalam ekstraksi teks (format slide atau hasil pindai). Klaim yang mengutip sumber ini mungkin ditandai TANPA SUMBER karena keterbatasan ekstraksi, bukan karena benar-benar tidak ada dalam sumber.`,

  // ── Results — claims ──
  contradictionsFound: 'Kontradiksi Ditemukan',
  allClaims: 'Semua Klaim',
  uncited: 'Tanpa Kutipan',
  citedClaimsHeading: 'Klaim Terkutip — Diperiksa Terhadap Sumber',
  uncitedClaimsHeading: 'Klaim Tanpa Kutipan',
  uncitedClaimsSubtitle: 'Tidak diperiksa terhadap sumber Anda — tinjau sendiri untuk memastikan ini adalah analisis Anda sendiri.',
  noCitation: 'Tanpa kutipan',
  markedAsYours: 'Ditandai sebagai milik Anda',
  thisIsMine: 'Ini milik saya',
  helpMeCiteThis: 'Bantu saya mengutip ini',
  closestMatch: 'Kecocokan terdekat dalam sumber Anda — verifikasi sendiri sebelum mengutip',
  sourcesChecked: 'Sumber yang diperiksa',

  statusSupported: 'Didukung',
  statusPartial: 'Sebagian',
  statusUnsupported: 'Tanpa sumber',
  statusContradicted: 'Bertentangan',
  unsupportedSourceNotUploaded: 'SUMBER TIDAK DIUNGGAH',
  unsupportedExtractionLimited: 'EKSTRAKSI TERBATAS',
  unsupportedNotFoundInSource: 'TIDAK DITEMUKAN DALAM SUMBER',
  categoryStatistic: 'Statistik tanpa kutipan',
  categoryComparative: 'Klaim perbandingan tanpa kutipan',
  categoryInterpretive: 'Klaim interpretatif yang ditandai oleh aturan',

  // ── Settings modal ──
  connection: 'Koneksi',
  provider: 'Penyedia',
  recommendedBadge: 'Direkomendasikan',
  browserProviderLabel: 'Lokal — tanpa kunci API, tanpa penyiapan',
  browserProviderDesc: 'Berjalan sepenuhnya di browser Anda. Makalah, sumber, dan hasil Anda tidak pernah meninggalkan perangkat Anda — tidak ada yang diunggah ke mana pun, tidak perlu akun, dan tetap berfungsi offline setelah pemuatan pertama.',
  localProviderLabel: 'AI Lokal (Ollama, LM Studio…)',
  localProviderDesc: 'Arahkan ke server model yang berjalan di perangkat Anda sendiri (misalnya Ollama) untuk penalaran setara LLM tanpa kunci API — memerlukan server tersebut aktif dan sedikit penyiapan.',
  bringYourOwnKey: 'Atau gunakan kunci API Anda sendiri (Claude, DeepSeek, OpenRouter…)',
  hide: 'Sembunyikan',
  providerName: 'Nama penyedia',
  providerNamePlaceholder: 'mis. claude, deepseek, openrouter…',
  notRecognizedNote: 'Tidak dikenali — tidak apa-apa, klik di luar dan isi sendiri URL dasar serta model di bawah.',
  baseUrl: 'URL Dasar',
  baseUrlPlaceholder: 'mis. https://api.example.com/v1',
  baseUrlLocalPlaceholder: 'mis. http://localhost:11434/v1',
  model: 'Model',
  modelPlaceholder: 'Nama model',
  apiKey: 'Kunci API',
  apiKeyPlaceholder: 'Kunci API',
  apiKeyOptionalPlaceholder: 'Opsional untuk server lokal',
  huntContradictionsLabel: 'Cari kontradiksi (tahap terpisah, ~2× panggilan)',
  storedOnlyNote: 'Hanya disimpan di browser ini (localStorage), hanya dikirim ke URL dasar di atas.',
  browserPipelineExplainer: 'Tanpa kunci API, tanpa akun, tanpa server. Verifikasi pertama Anda memerlukan internet satu kali, untuk mengunduh dua model kecil ke browser ini (setelah itu tersimpan dalam cache) — setiap verifikasi berikutnya berjalan sepenuhnya offline, kecuali Anda beralih ke penyedia berbasis kunci API di bawah, yang selalu memerlukan koneksi. Kualitasnya lebih terbatas dibandingkan penyedia berbasis LLM (deteksi klaim berbasis aturan, penalaran embedding + NLI, bukan pemahaman bahasa penuh), tetapi tidak ada biaya dan tidak perlu konfigurasi. Pengaturan bawaan di bawah sudah merupakan opsi dengan hasil pengukuran terbaik (hasil uji nyata, bukan perkiraan) — kebanyakan orang tidak perlu membuka "Lanjutan".',
  advancedModelSettings: 'Pengaturan model lanjutan',
  embeddingModelRetrieval: 'Model embedding (pencarian)',
  nliModelReasoning: 'Model NLI (penalaran)',
  retrievalMethod: 'Metode pencarian',
  qualityOnTestSet: pct => `${pct}% pada set uji kami`,
  done: 'Selesai',

  // ── Footer ──
  privacyParagraph: 'Semuanya berjalan di browser Anda sendiri. Makalah, sumber, dan kunci API Anda tidak pernah dikirim ke atau disimpan di server mana pun yang kami operasikan — karena kami tidak mengoperasikan server. Kunci Anda langsung dikirim dari browser Anda ke penyedia AI pilihan Anda, dan kami tidak pernah melihat atau menyimpan salinannya. Anda bertanggung jawab atas penggunaan, biaya, dan ketentuan penyedia tersebut.',
  scilenePromoIntro: 'Setelah kutipan Anda terverifikasi, pertanyaan berikutnya biasanya ke mana harus mengirimkan naskah.',
  scilenePromoBody: '— alat pendamping dari penulis yang sama — mencocokkan naskah Anda dengan jurnal yang terindeks di Scopus, Web of Science, DOAJ, dan SINTA, dengan penjelasan sederhana mengapa masing-masing cocok. Pendekatan yang sama seperti di sini: tanpa akun, berjalan secara lokal.',
  sourceOnGithub: 'Sumber di GitHub',
  builtBy: 'Dibuat oleh',

  // ── Errors / validation ──
  enterApiKey: label => `Masukkan kunci API ${label} Anda.`,
  enterBaseUrl: 'Masukkan URL dasar API.',
  enterModelName: 'Masukkan nama model.',
  noPaperText: 'Tidak ada teks makalah untuk diverifikasi.',
  uploadAtLeastOneSource: 'Unggah setidaknya satu sumber referensi.',
  couldNotReadPaper: msg => `Tidak dapat membaca makalah: ${msg}`,
  couldNotReadReference: msg => `Tidak dapat membaca referensi: ${msg}`,
  couldNotReadProgressFile: msg => `Tidak dapat membaca file progres: ${msg}`,
  verificationFailed: msg => `Verifikasi gagal: ${msg}`,
  resumeWrongPipeline: 'Proses yang dijeda ini berasal dari verifikasi penyedia AI, bukan alur browser lokal — beralihlah ke penyedia yang Anda gunakan sebelumnya (melalui ikon pengaturan) lalu klik Lanjutkan lagi.',
  stoppedAround: (batchLabel, total, msg) => `Berhenti di sekitar ${batchLabel} dari ${total}: ${msg}. Progres telah disimpan — klik Lanjutkan untuk melanjutkan, atau unduh di bawah.`,

  // ── Run-status templates ──
  chunkLabel: i => `bagian ${i}`,
  chunksLabel: (i, j) => `bagian ${i}–${j}`,
  checkingBatch: (batchLabel, total) => `Memeriksa ${batchLabel} dari ${total}…`,
  checkingBatchRetrying: (batchLabel, total, attempt, max) => `Memeriksa ${batchLabel} dari ${total} — respons tidak dapat digunakan, mencoba lagi (${attempt}/${max})…`,
  responseRetrying: (attempt, max) => `Respons tidak dapat digunakan, mencoba lagi (${attempt}/${max})…`,
  checkingAndHunting: 'Memeriksa klaim dan mencari kontradiksi…',
  loadingLocalModels: 'Memuat model lokal (proses pertama akan mengunduhnya, setelah itu tersimpan dalam cache)…',
  indexingSentences: (n, sources) => `Mengindeks ${n} kalimat dari ${sources} sumber…`,
  indexingProgress: (done, total) => `Mengindeks kalimat ${done} dari ${total}…`,
  checkingCitedClaims: n => `Memeriksa ${n} klaim terkutip…`,
  checkingClaimProgress: (done, total) => `Memeriksa klaim ${done} dari ${total}…`,

  notFoundInSources: 'tidak ditemukan dalam sumber',
  noSimilarPassage: 'Tidak ditemukan bagian yang cukup mirip dalam sumber.',
  flaggedByLocalRules: 'Ditandai oleh aturan lokal (angka, perbandingan, atau kata kerja penalaran) — tidak ada kutipan yang melekat.',
  liveChecked: 'Diperiksa',
  liveSupported: 'Didukung',
  livePartial: 'Sebagian',
  liveIssues: 'Masalah',
  latestChecked: 'Terakhir diperiksa',

  // ── Report summary (buildFinalResult/summarize) ──
  noCitedClaims: n => `Tidak ada klaim terkutip yang ditemukan untuk diperiksa. ${n} kalimat ditandai sebagai berpotensi tanpa kutipan setelah penyaringan.`,
  reportSummary: ({ total, supported, partial, unsupported, sourceNotUploaded, extractionLimited, notFoundInSource, contradicted, uncitedCount }) =>
    `${total} klaim terkutip diperiksa. ${supported} didukung, ${partial} sebagian, ${unsupported} tanpa sumber `
    + `(${sourceNotUploaded} karena sumber tidak ada, ${extractionLimited} karena keterbatasan ekstraksi, ${notFoundInSource} benar-benar tidak ditemukan)`
    + `${contradicted ? `, ${contradicted} bertentangan` : ''}. ${uncitedCount} kalimat ditandai sebagai berpotensi tanpa kutipan setelah penyaringan.`,

  // ── Downloadable/copyable report text (buildReportText) ──
  reportTitle: 'LOUPE — LAPORAN VERIFIKASI SUMBER',
  reportScoreLine: (raw, adjusted) => `Skor mentah: ${raw}/100   Skor disesuaikan: ${adjusted}/100`,
  reportCoverageLine: (found, total) => `Cakupan sumber: ${found} dari ${total} sumber yang dikutip ditemukan dalam korpus`,
  reportExtractionWarningsLine: names => `Sumber dengan peringatan ekstraksi: ${names}`,
  reportReferencesLine: n => `Bagian referensi: ${n} entri teridentifikasi dan dikecualikan dari verifikasi.`,
  reportSourceCoverageHeading: 'CAKUPAN SUMBER',
  reportMissingSourcesLine: names => `Sumber yang dikutip berikut tidak ditemukan dalam korpus verifikasi: ${names}.`,
  reportUploadBeforeRerun: 'Unggah sumber-sumber ini sebelum menjalankan ulang untuk mendapatkan skor yang andal.',
  reportContradictionsHeading: 'KONTRADIKSI',
  reportCitedClaimsHeading: 'KLAIM TERKUTIP',
  reportEvidenceLine: q => `Bukti: "${q}"`,
  reportSourceLine: name => `Sumber: ${name}`,
  reportUncitedHeading: 'KLAIM TANPA KUTIPAN',
};

export const STRINGS = { en, id };

export function stringsFor(lang) {
  return STRINGS[lang] || STRINGS.en;
}
