// Indonesian -> English academic/methodology glossary used to translate
// Indonesian claim/evidence text into English before it reaches the
// English-tuned embedding and NLI models (see inference.worker.js).
//
// Hand-curated rather than machine-generated: MT quality on isolated short
// terms (no sentence context to disambiguate) is a known weak point
// compared to MT's strength on full sentences — "signifikan" round-tripped
// through a generic translator can just as easily land on "important" as
// "significant", and a term-substitution dictionary has no sentence context
// to correct that. Terminology glossaries are conventionally built by a
// bilingual curator for exactly this reason; MT is used here for full-
// sentence translation at runtime instead (opus-mt-id-en), where it's
// actually the right tool.
//
// Deliberately scoped to academic/scientific register — general-vocabulary
// coverage isn't the goal, matching a claim/evidence sentence's topic well
// enough for retrieval (and, if the benchmark in bench/run.mjs favors it,
// for the NLI leg too) is.
//
// English side is deliberately plain/generic phrasing (not literary
// synonyms) since it's feeding an embedding/NLI model, not a human reader.
const ENTRIES = {
  // Research process / methodology
  'penelitian': 'research', 'riset': 'research', 'studi': 'study',
  'kajian': 'study', 'analisis': 'analysis', 'menganalisis': 'analyze',
  'metode': 'method', 'metodologi': 'methodology', 'pendekatan': 'approach',
  'data': 'data', 'dataset': 'dataset', 'sampel': 'sample',
  'populasi': 'population', 'responden': 'respondents',
  'partisipan': 'participants', 'peserta': 'participants',
  'variabel': 'variable', 'indikator': 'indicator', 'parameter': 'parameter',
  'hipotesis': 'hypothesis', 'kerangka': 'framework', 'teori': 'theory',
  'model': 'model', 'konsep': 'concept', 'definisi': 'definition',
  'survei': 'survey', 'wawancara': 'interview',
  'kuesioner': 'questionnaire', 'angket': 'questionnaire',
  'observasi': 'observation', 'eksperimen': 'experiment',
  'uji': 'test', 'pengujian': 'testing', 'pengukuran': 'measurement',
  'instrumen': 'instrument', 'validitas': 'validity',
  'reliabilitas': 'reliability', 'keabsahan': 'validity',
  'literatur': 'literature', 'pustaka': 'literature',
  'tinjauan': 'review', 'kepustakaan': 'literature',

  // Findings / reporting verbs
  'temuan': 'finding', 'hasil': 'result', 'kesimpulan': 'conclusion',
  'menyimpulkan': 'concludes', 'menunjukkan': 'shows',
  'memperlihatkan': 'shows', 'membuktikan': 'proves',
  'mengonfirmasi': 'confirms', 'mengungkapkan': 'reveals',
  'ditemukan': 'found', 'menemukan': 'finds',
  'melaporkan': 'reports', 'dilaporkan': 'reported',
  'mendokumentasikan': 'documents', 'mengidentifikasi': 'identifies',
  'mengamati': 'observes', 'teramati': 'observed',
  'menyatakan': 'states', 'berpendapat': 'argues',
  'berargumen': 'argues', 'menegaskan': 'confirms',
  'mengindikasikan': 'indicates', 'mencatat': 'notes',
  'dicatat': 'noted', 'menyoroti': 'highlights',

  // Comparatives / statistics
  'lebih': 'more', 'kurang': 'less', 'daripada': 'than',
  'dibandingkan': 'compared', 'perbandingan': 'comparison',
  'meningkat': 'increased', 'peningkatan': 'increase',
  'menurun': 'decreased', 'penurunan': 'decrease',
  'bertambah': 'increased', 'berkurang': 'decreased',
  'meningkatkan': 'increases', 'menurunkan': 'lowers', 'mengurangi': 'reduces',
  'dikurangi': 'reduced', 'ditingkatkan': 'increased',
  'signifikan': 'significant', 'signifikansi': 'significance',
  'secara': 'significantly', 'tinggi': 'higher', 'rendah': 'lower',
  'besar': 'large', 'kecil': 'small', 'rata-rata': 'average',
  'persentase': 'percentage', 'persen': 'percent', 'jumlah': 'number',
  'total': 'total', 'mayoritas': 'majority', 'minoritas': 'minority',
  'proporsi': 'proportion', 'tingkat': 'rate', 'skor': 'score',
  'nilai': 'value', 'rasio': 'ratio', 'median': 'median',

  // Causation / correlation
  'pengaruh': 'influence', 'memengaruhi': 'influences',
  'dipengaruhi': 'influenced', 'dampak': 'impact', 'berdampak': 'impacts',
  'efek': 'effect', 'faktor': 'factor', 'penyebab': 'cause',
  'menyebabkan': 'causes', 'disebabkan': 'caused',
  'korelasi': 'correlation', 'berkorelasi': 'correlates',
  'hubungan': 'relationship', 'berhubungan': 'related',
  'terkait': 'related', 'kaitan': 'connection',

  // Structure / outcomes / policy
  'kebijakan': 'policy', 'strategi': 'strategy',
  'implementasi': 'implementation', 'menerapkan': 'implements',
  'diterapkan': 'implemented', 'program': 'program',
  'intervensi': 'intervention', 'evaluasi': 'evaluation',
  'mengevaluasi': 'evaluates', 'efektivitas': 'effectiveness',
  'efektif': 'effective', 'inefisien': 'inefficient',
  'kualitas': 'quality', 'kuantitas': 'quantity',
  'kapasitas': 'capacity', 'kinerja': 'performance',
  'akses': 'access', 'partisipasi': 'participation',
  'keterlibatan': 'involvement', 'transparansi': 'transparency',
  'akuntabilitas': 'accountability', 'kepatuhan': 'compliance',
  'peran': 'role', 'fungsi': 'function', 'tujuan': 'objective',
  'sasaran': 'target', 'target': 'target', 'manfaat': 'benefit',
  'keuntungan': 'benefit', 'kerugian': 'loss', 'risiko': 'risk',
  'tantangan': 'challenge', 'hambatan': 'obstacle', 'kendala': 'constraint',
  'permasalahan': 'problem', 'masalah': 'problem', 'isu': 'issue',
  'solusi': 'solution', 'rekomendasi': 'recommendation',

  // Clinical/medical vocabulary — added after the first benchmark run
  // (bench/run.mjs, 2026-08-30) measured only 50% retrieval accuracy on
  // ID_RETRIEVAL_TESTSET and tracing the misses back to exactly this gap:
  // the academic/methodology register above doesn't cover a clinical-trial
  // paper's actual content words ("gejala", "vaksin", "tekanan darah"),
  // so those terms passed through untranslated and hurt cosine similarity
  // against the English evidence pool. Not exhaustive — added to cover
  // this test set's real misses, the kind of domain-vocabulary gap that
  // widens over time with real use, not a closed set.
  'pengobatan': 'treatment', 'perawatan': 'treatment', 'diobati': 'treated',
  'obat': 'drug', 'dosis': 'dose', 'gejala': 'symptom', 'penyakit': 'disease',
  'pasien': 'patient', 'rumah sakit': 'hospital', 'klinis': 'clinical',
  'uji klinis': 'clinical trial', 'plasebo': 'placebo',
  'efek samping': 'side effect', 'tekanan darah': 'blood pressure',
  'vaksin': 'vaccine', 'divaksinasi': 'vaccinated', 'penularan': 'transmission',
  'menular': 'transmitted', 'kecemasan': 'anxiety', 'gangguan': 'disorder',
  'tumor': 'tumor', 'senyawa': 'compound', 'kelompok': 'group',
  'sesi': 'session', 'keparahan': 'severity',
  'skala': 'scale', 'klasifikasi': 'classification', 'akurasi': 'accuracy',
  'pelatihan': 'training', 'algoritma': 'algorithm', 'diagnosis': 'diagnosis',
  'mengklasifikasikan': 'classifies', 'diklasifikasikan': 'classified',
  'benar': 'correct', 'sekitar': 'approximately', 'contoh': 'example',
  'mengalami': 'experienced', 'pertumbuhan': 'growth', 'penjualan': 'sales',
  'tumbuh': 'grew',

  // General academic register
  'penting': 'important', 'utama': 'main', 'umum': 'general',
  'khusus': 'specific', 'spesifik': 'specific', 'baru': 'new',
  'terbaru': 'recent', 'sebelumnya': 'previous', 'saat ini': 'current',
  'berbagai': 'various', 'beberapa': 'several', 'setiap': 'each',
  'seluruh': 'entire', 'keseluruhan': 'overall', 'sebagian': 'partial',
  'kompleks': 'complex', 'sederhana': 'simple', 'jelas': 'clear',
  'relevan': 'relevant', 'konsisten': 'consistent', 'konteks': 'context',
  'perspektif': 'perspective', 'aspek': 'aspect', 'dimensi': 'dimension',
  'kategori': 'category', 'kriteria': 'criteria', 'karakteristik': 'characteristic',

  // Common connectors (also useful for the en/id source-language detector)
  'yang': 'that', 'dan': 'and', 'atau': 'or', 'tidak': 'not',
  'adalah': 'is', 'dari': 'from', 'dengan': 'with', 'ini': 'this',
  'itu': 'that', 'untuk': 'for', 'pada': 'on', 'dalam': 'in',
  'akan': 'will', 'telah': 'has', 'oleh': 'by', 'sebagai': 'as',
  'karena': 'because', 'sehingga': 'so that', 'namun': 'however',
  'tetapi': 'but', 'sedangkan': 'whereas', 'sementara': 'while',
  'juga': 'also', 'hanya': 'only', 'masih': 'still', 'sudah': 'already',
  'antara': 'between', 'terhadap': 'towards', 'melalui': 'through',
  'berdasarkan': 'based on', 'menurut': 'according to', 'sesuai': 'according to',

  // Time / place
  'tahun': 'year', 'bulan': 'month', 'periode': 'period',
  'waktu': 'time', 'wilayah': 'region', 'daerah': 'area',
  'negara': 'country', 'kota': 'city', 'nasional': 'national',
  'lokal': 'local', 'internasional': 'international', 'global': 'global',

  // Domain nouns that recur across social-science/health/tech papers
  'masyarakat': 'community', 'pemerintah': 'government',
  'lembaga': 'institution', 'organisasi': 'organization',
  'institusi': 'institution', 'perusahaan': 'company',
  'industri': 'industry', 'ekonomi': 'economy', 'sosial': 'social',
  'budaya': 'culture', 'pendidikan': 'education', 'kesehatan': 'health',
  'lingkungan': 'environment', 'teknologi': 'technology',
  'digital': 'digital', 'sistem': 'system', 'proses': 'process',
  'layanan': 'service', 'pelayanan': 'service', 'infrastruktur': 'infrastructure',
};

// Longest-match-first substitution avoids "menurut" (according to) getting
// clobbered by a shorter unrelated entry that happens to be a substring,
// and handles multi-word entries ("saat ini", "berdasarkan") correctly.
const SORTED_TERMS = Object.keys(ENTRIES).sort((a, b) => b.length - a.length);
const LOOKUP_PATTERN = new RegExp(
  `\\b(${SORTED_TERMS.map(t => t.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|')})\\b`,
  'gi'
);

// Replaces every recognized Indonesian term in `text` with its English
// equivalent, leaving unrecognized words (names, numbers, domain-specific
// terms outside this glossary) untouched — a partial translation is still
// far better input to an English-tuned embedding/NLI model than none, per
// the measured comparison this dictionary approach is based on.
export function translateIdToEn(text) {
  return text.replace(LOOKUP_PATTERN, match => ENTRIES[match.toLowerCase()] || match);
}

export const ID_EN_DICTIONARY = ENTRIES;
