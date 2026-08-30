// A small hand-written test set covering the verdict types and known failure
// modes for citation-verification NLI (direct match, paraphrase, numeric
// match/mismatch, partial overlap, negation, absence, misleading similarity).
// Not a rigorous academic benchmark — just enough real signal to rank the
// model options honestly instead of guessing at percentages.

export const NLI_TESTSET = [
  // Direct match
  { evidence: "The treatment reduced symptoms by 40% in the treated group.", claim: "The treatment reduced symptoms by 40%.", expected: "SUPPORTED" },
  { evidence: "Researchers observed that mice exposed to the compound showed reduced tumor growth.", claim: "The compound reduced tumor growth in mice.", expected: "SUPPORTED" },
  // Paraphrase / synonym match
  { evidence: "Participants who received the intervention reported a marked decline in anxiety scores.", claim: "The intervention lowered anxiety in participants.", expected: "SUPPORTED" },
  { evidence: "The algorithm achieved a classification accuracy of 94.2% on the held-out test set.", claim: "The model correctly classified about 94% of test examples.", expected: "SUPPORTED" },
  // Numeric match
  { evidence: "Enrollment reached 312 participants across five sites by the end of 2022.", claim: "The study enrolled 312 participants.", expected: "SUPPORTED" },
  // Numeric mismatch (subtle, should NOT be marked supported)
  { evidence: "The drug lowered blood pressure by 12 mmHg on average.", claim: "The drug lowered blood pressure by 20 mmHg on average.", expected: "CONTRADICTED" },
  // Partial overlap
  { evidence: "The program improved reading scores among younger children, though effects for older students were unclear.", claim: "The program improved reading scores across all age groups.", expected: "PARTIAL" },
  { evidence: "Sales grew in North America but declined in every other region.", claim: "The company saw global sales growth.", expected: "PARTIAL" },
  // Negation / contradiction
  { evidence: "No significant difference in survival rate was observed between the two groups.", claim: "The treatment significantly improved survival rate.", expected: "CONTRADICTED" },
  { evidence: "The follow-up study found the original result did not replicate.", claim: "The original finding was later confirmed by follow-up research.", expected: "CONTRADICTED" },
  // Complete absence (evidence about something else entirely)
  { evidence: "The manufacturing facility relocated to a larger site in 2019 to increase production capacity.", claim: "The treatment reduced symptoms by 40%.", expected: "UNSUPPORTED" },
  { evidence: "Coffee consumption has increased steadily across urban populations over the past decade.", claim: "The algorithm achieved 94% accuracy.", expected: "UNSUPPORTED" },
  // Misleading similarity — shares vocabulary but opposite meaning
  { evidence: "The vaccine showed no measurable reduction in transmission rates among vaccinated individuals.", claim: "The vaccine reduced transmission rates.", expected: "CONTRADICTED" },
  { evidence: "Critics argue the policy increased inequality rather than reducing it, despite its stated goals.", claim: "The policy reduced inequality, as intended.", expected: "CONTRADICTED" },
];

// For the embedding/retrieval benchmark: each item is a claim plus a small
// pool of candidate sentences, exactly one of which is the true match — the
// rest are plausible-looking distractors sharing vocabulary or topic.
export const RETRIEVAL_TESTSET = [
  {
    claim: "The treatment reduced symptoms by 40% in the treated group.",
    correctIndex: 0,
    pool: [
      "In our 2021 clinical trial, patients receiving the treatment showed a 40 percent reduction in reported symptoms.",
      "The placebo group showed a 5% reduction in symptoms over the same period.",
      "The treatment was administered twice daily for eight weeks.",
      "Symptom severity was measured using a standardized 10-point clinical scale.",
    ],
  },
  {
    claim: "The model correctly classified about 94% of test examples.",
    correctIndex: 2,
    pool: [
      "The dataset contained 10,000 labeled training examples across twelve categories.",
      "Training took approximately six hours on a single GPU.",
      "On the held-out test set, the classifier achieved 94.2% accuracy.",
      "A baseline logistic regression model achieved 71% accuracy on the same task.",
    ],
  },
  {
    claim: "The company saw global sales growth.",
    correctIndex: 1,
    pool: [
      "The company's headquarters moved to a new office in downtown Seattle.",
      "Sales grew 8% in North America but declined 3% in Europe and 6% in Asia.",
      "The CEO announced a new product line at the annual shareholder meeting.",
      "Employee headcount increased by 200 over the fiscal year.",
    ],
  },
  {
    claim: "No significant difference in survival rate was found between treatment groups.",
    correctIndex: 3,
    pool: [
      "Patients in the treatment group reported fewer side effects than the control group.",
      "The trial was conducted across fourteen hospitals in three countries.",
      "Median follow-up time was 18 months for both cohorts.",
      "Survival analysis showed no statistically significant difference between the treatment and control arms (p=0.42).",
    ],
  },
  {
    claim: "The vaccine reduced transmission rates.",
    correctIndex: 0,
    pool: [
      "Transmission rates among vaccinated individuals were statistically indistinguishable from unvaccinated individuals.",
      "The vaccine was approved for emergency use in early 2021.",
      "Manufacturing capacity was scaled up to meet global demand.",
      "Side effects were generally mild and resolved within 48 hours.",
    ],
  },
  {
    claim: "The intervention lowered anxiety in participants.",
    correctIndex: 3,
    pool: [
      "The intervention consisted of eight weekly group sessions.",
      "Participants were recruited from three community health centers.",
      "Dropout rate was 12% over the course of the study.",
      "Anxiety scores on the standardized inventory dropped by an average of 6.3 points post-intervention, a marked decline from baseline.",
    ],
  },
];

// Indonesian translations of NLI_TESTSET, same order, same expected verdicts
// — used to compare candidate translation strategies (dictionary
// substitution vs. full MT vs. a multilingual NLI model with no
// translation) against the exact same verdict distribution the English
// numbers above were measured on. See src/lib/dictionary.id-en.js for the
// dictionary itself.
export const ID_NLI_TESTSET = [
  { evidence: "Pengobatan tersebut mengurangi gejala sebesar 40% pada kelompok yang diobati.", claim: "Pengobatan tersebut mengurangi gejala sebesar 40%.", expected: "SUPPORTED" },
  { evidence: "Para peneliti mengamati bahwa tikus yang terpapar senyawa tersebut menunjukkan penurunan pertumbuhan tumor.", claim: "Senyawa tersebut mengurangi pertumbuhan tumor pada tikus.", expected: "SUPPORTED" },
  { evidence: "Peserta yang menerima intervensi melaporkan penurunan signifikan pada skor kecemasan.", claim: "Intervensi tersebut menurunkan kecemasan pada peserta.", expected: "SUPPORTED" },
  { evidence: "Algoritma tersebut mencapai akurasi klasifikasi sebesar 94,2% pada set pengujian yang terpisah.", claim: "Model tersebut mengklasifikasikan dengan benar sekitar 94% dari contoh pengujian.", expected: "SUPPORTED" },
  { evidence: "Jumlah pendaftaran mencapai 312 peserta di lima lokasi pada akhir tahun 2022.", claim: "Studi tersebut melibatkan 312 peserta.", expected: "SUPPORTED" },
  { evidence: "Obat tersebut menurunkan tekanan darah rata-rata sebesar 12 mmHg.", claim: "Obat tersebut menurunkan tekanan darah rata-rata sebesar 20 mmHg.", expected: "CONTRADICTED" },
  { evidence: "Program tersebut meningkatkan skor membaca pada anak-anak yang lebih muda, meskipun efeknya belum jelas untuk siswa yang lebih tua.", claim: "Program tersebut meningkatkan skor membaca di semua kelompok usia.", expected: "PARTIAL" },
  { evidence: "Penjualan tumbuh di Amerika Utara tetapi menurun di setiap wilayah lainnya.", claim: "Perusahaan tersebut mengalami pertumbuhan penjualan global.", expected: "PARTIAL" },
  { evidence: "Tidak ditemukan perbedaan signifikan dalam tingkat kelangsungan hidup antara kedua kelompok.", claim: "Pengobatan tersebut secara signifikan meningkatkan tingkat kelangsungan hidup.", expected: "CONTRADICTED" },
  { evidence: "Studi lanjutan menemukan bahwa hasil awal tidak dapat direplikasi.", claim: "Temuan awal tersebut kemudian dikonfirmasi oleh penelitian lanjutan.", expected: "CONTRADICTED" },
  { evidence: "Fasilitas manufaktur tersebut pindah ke lokasi yang lebih besar pada tahun 2019 untuk meningkatkan kapasitas produksi.", claim: "Pengobatan tersebut mengurangi gejala sebesar 40%.", expected: "UNSUPPORTED" },
  { evidence: "Konsumsi kopi telah meningkat secara stabil di kalangan penduduk perkotaan selama satu dekade terakhir.", claim: "Algoritma tersebut mencapai akurasi 94%.", expected: "UNSUPPORTED" },
  { evidence: "Vaksin tersebut tidak menunjukkan penurunan tingkat penularan yang terukur pada individu yang divaksinasi.", claim: "Vaksin tersebut menurunkan tingkat penularan.", expected: "CONTRADICTED" },
  { evidence: "Para kritikus berpendapat bahwa kebijakan tersebut justru meningkatkan ketimpangan alih-alih menguranginya, meskipun tujuannya dinyatakan berbeda.", claim: "Kebijakan tersebut mengurangi ketimpangan, sesuai dengan yang dimaksudkan.", expected: "CONTRADICTED" },
];

export const ID_RETRIEVAL_TESTSET = [
  {
    claim: "Pengobatan tersebut mengurangi gejala sebesar 40% pada kelompok yang diobati.",
    correctIndex: 0,
    pool: [
      "Dalam uji klinis kami tahun 2021, pasien yang menerima pengobatan menunjukkan penurunan gejala yang dilaporkan sebesar 40 persen.",
      "Kelompok plasebo menunjukkan penurunan gejala sebesar 5% selama periode yang sama.",
      "Pengobatan diberikan dua kali sehari selama delapan minggu.",
      "Keparahan gejala diukur menggunakan skala klinis standar 10 poin.",
    ],
  },
  {
    claim: "Model tersebut mengklasifikasikan dengan benar sekitar 94% dari contoh pengujian.",
    correctIndex: 2,
    pool: [
      "Dataset tersebut berisi 10.000 contoh pelatihan berlabel di dua belas kategori.",
      "Pelatihan memakan waktu sekitar enam jam pada satu GPU.",
      "Pada set pengujian yang terpisah, pengklasifikasi mencapai akurasi 94,2%.",
      "Model regresi logistik dasar mencapai akurasi 71% pada tugas yang sama.",
    ],
  },
  {
    claim: "Perusahaan tersebut mengalami pertumbuhan penjualan global.",
    correctIndex: 1,
    pool: [
      "Kantor pusat perusahaan pindah ke kantor baru di pusat kota Seattle.",
      "Penjualan tumbuh 8% di Amerika Utara tetapi menurun 3% di Eropa dan 6% di Asia.",
      "CEO mengumumkan lini produk baru pada rapat pemegang saham tahunan.",
      "Jumlah karyawan meningkat sebanyak 200 orang selama tahun fiskal.",
    ],
  },
  {
    claim: "Tidak ditemukan perbedaan signifikan dalam tingkat kelangsungan hidup antara kelompok pengobatan.",
    correctIndex: 3,
    pool: [
      "Pasien dalam kelompok pengobatan melaporkan efek samping yang lebih sedikit dibandingkan kelompok kontrol.",
      "Uji coba dilakukan di empat belas rumah sakit di tiga negara.",
      "Median waktu tindak lanjut adalah 18 bulan untuk kedua kohort.",
      "Analisis kelangsungan hidup menunjukkan tidak ada perbedaan yang signifikan secara statistik antara kelompok pengobatan dan kontrol (p=0,42).",
    ],
  },
  {
    claim: "Vaksin tersebut menurunkan tingkat penularan.",
    correctIndex: 0,
    pool: [
      "Tingkat penularan pada individu yang divaksinasi secara statistik tidak berbeda dari individu yang tidak divaksinasi.",
      "Vaksin tersebut disetujui untuk penggunaan darurat pada awal tahun 2021.",
      "Kapasitas produksi ditingkatkan untuk memenuhi permintaan global.",
      "Efek samping umumnya ringan dan hilang dalam 48 jam.",
    ],
  },
  {
    claim: "Intervensi tersebut menurunkan kecemasan pada peserta.",
    correctIndex: 3,
    pool: [
      "Intervensi tersebut terdiri dari delapan sesi kelompok mingguan.",
      "Peserta direkrut dari tiga pusat kesehatan masyarakat.",
      "Tingkat drop-out adalah 12% selama studi berlangsung.",
      "Skor kecemasan pada inventori standar turun rata-rata 6,3 poin setelah intervensi, penurunan yang signifikan dari kondisi awal.",
    ],
  },
];

// Arabic translations of the same cases, same purpose — see the Indonesian
// set above for the methodology note (exact same verdict distribution as
// NLI_TESTSET/RETRIEVAL_TESTSET, so the numbers are directly comparable).
export const AR_NLI_TESTSET = [
  { evidence: "العلاج قلل الأعراض بنسبة 40% في المجموعة المعالجة.", claim: "العلاج قلل الأعراض بنسبة 40%.", expected: "SUPPORTED" },
  { evidence: "لاحظ الباحثون أن الفئران المعرضة للمركب أظهرت انخفاضًا في نمو الورم.", claim: "قلل المركب من نمو الورم في الفئران.", expected: "SUPPORTED" },
  { evidence: "أفاد المشاركون الذين تلقوا التدخل بانخفاض ملحوظ في درجات القلق.", claim: "قلل التدخل من القلق لدى المشاركين.", expected: "SUPPORTED" },
  { evidence: "حققت الخوارزمية دقة تصنيف بلغت 94.2% على مجموعة الاختبار المستقلة.", claim: "صنف النموذج بشكل صحيح حوالي 94% من أمثلة الاختبار.", expected: "SUPPORTED" },
  { evidence: "بلغ عدد المسجلين 312 مشاركًا عبر خمسة مواقع بحلول نهاية عام 2022.", claim: "شملت الدراسة 312 مشاركًا.", expected: "SUPPORTED" },
  { evidence: "خفض الدواء ضغط الدم بمعدل 12 ملم زئبقي في المتوسط.", claim: "خفض الدواء ضغط الدم بمعدل 20 ملم زئبقي في المتوسط.", expected: "CONTRADICTED" },
  { evidence: "حسّن البرنامج درجات القراءة لدى الأطفال الأصغر سنًا، رغم أن تأثيره على الطلاب الأكبر سنًا لم يكن واضحًا.", claim: "حسّن البرنامج درجات القراءة في جميع الفئات العمرية.", expected: "PARTIAL" },
  { evidence: "نمت المبيعات في أمريكا الشمالية لكنها انخفضت في كل منطقة أخرى.", claim: "شهدت الشركة نموًا عالميًا في المبيعات.", expected: "PARTIAL" },
  { evidence: "لم يُلاحظ أي فرق ذي دلالة إحصائية في معدل البقاء على قيد الحياة بين المجموعتين.", claim: "حسّن العلاج معدل البقاء على قيد الحياة بشكل ملحوظ.", expected: "CONTRADICTED" },
  { evidence: "وجدت دراسة المتابعة أن النتيجة الأصلية لم تتكرر.", claim: "تم تأكيد النتيجة الأصلية لاحقًا من خلال أبحاث المتابعة.", expected: "CONTRADICTED" },
  { evidence: "انتقل مرفق التصنيع إلى موقع أكبر في عام 2019 لزيادة القدرة الإنتاجية.", claim: "العلاج قلل الأعراض بنسبة 40%.", expected: "UNSUPPORTED" },
  { evidence: "ازداد استهلاك القهوة بشكل مطرد بين سكان المدن خلال العقد الماضي.", claim: "حققت الخوارزمية دقة 94%.", expected: "UNSUPPORTED" },
  { evidence: "لم يُظهر اللقاح أي انخفاض ملموس في معدلات انتقال العدوى بين الأفراد الملقحين.", claim: "خفض اللقاح معدلات انتقال العدوى.", expected: "CONTRADICTED" },
  { evidence: "يجادل النقاد بأن السياسة زادت من عدم المساواة بدلاً من الحد منها، رغم أهدافها المعلنة.", claim: "قللت السياسة من عدم المساواة كما كان مقصودًا.", expected: "CONTRADICTED" },
];

export const AR_RETRIEVAL_TESTSET = [
  {
    claim: "العلاج قلل الأعراض بنسبة 40% في المجموعة المعالجة.",
    correctIndex: 0,
    pool: [
      "في تجربتنا السريرية لعام 2021، أظهر المرضى الذين تلقوا العلاج انخفاضًا بنسبة 40 بالمئة في الأعراض المبلغ عنها.",
      "أظهرت مجموعة الدواء الوهمي انخفاضًا بنسبة 5% في الأعراض خلال نفس الفترة.",
      "أُعطي العلاج مرتين يوميًا لمدة ثمانية أسابيع.",
      "قِيست شدة الأعراض باستخدام مقياس سريري موحد من 10 نقاط.",
    ],
  },
  {
    claim: "صنف النموذج بشكل صحيح حوالي 94% من أمثلة الاختبار.",
    correctIndex: 2,
    pool: [
      "احتوت مجموعة البيانات على 10,000 مثال تدريب موسوم عبر اثنتي عشرة فئة.",
      "استغرق التدريب حوالي ست ساعات على وحدة معالجة رسومات واحدة.",
      "على مجموعة الاختبار المستقلة، حقق المصنف دقة بلغت 94.2%.",
      "حقق نموذج انحدار لوجستي أساسي دقة 71% على نفس المهمة.",
    ],
  },
  {
    claim: "شهدت الشركة نموًا عالميًا في المبيعات.",
    correctIndex: 1,
    pool: [
      "انتقل المقر الرئيسي للشركة إلى مكتب جديد في وسط مدينة سياتل.",
      "نمت المبيعات بنسبة 8% في أمريكا الشمالية لكنها انخفضت بنسبة 3% في أوروبا و6% في آسيا.",
      "أعلن الرئيس التنفيذي عن خط إنتاج جديد في اجتماع المساهمين السنوي.",
      "ازداد عدد الموظفين بمقدار 200 خلال السنة المالية.",
    ],
  },
  {
    claim: "لم يُلاحظ فرق ذو دلالة إحصائية في معدل البقاء على قيد الحياة بين مجموعات العلاج.",
    correctIndex: 3,
    pool: [
      "أفاد المرضى في مجموعة العلاج بآثار جانبية أقل مقارنة بمجموعة التحكم.",
      "أُجريت التجربة عبر أربعة عشر مستشفى في ثلاث دول.",
      "كان متوسط مدة المتابعة 18 شهرًا لكلتا المجموعتين.",
      "أظهر تحليل البقاء على قيد الحياة عدم وجود فرق ذي دلالة إحصائية بين مجموعتي العلاج والتحكم (p=0.42).",
    ],
  },
  {
    claim: "خفض اللقاح معدلات انتقال العدوى.",
    correctIndex: 0,
    pool: [
      "كانت معدلات انتقال العدوى بين الأفراد الملقحين لا تختلف إحصائيًا عن غير الملقحين.",
      "تمت الموافقة على اللقاح للاستخدام الطارئ في أوائل عام 2021.",
      "تم توسيع القدرة الإنتاجية لتلبية الطلب العالمي.",
      "كانت الآثار الجانبية خفيفة بشكل عام وزالت خلال 48 ساعة.",
    ],
  },
  {
    claim: "قلل التدخل من القلق لدى المشاركين.",
    correctIndex: 3,
    pool: [
      "تألف التدخل من ثماني جلسات جماعية أسبوعية.",
      "تم تجنيد المشاركين من ثلاثة مراكز صحية مجتمعية.",
      "كان معدل الانسحاب 12% خلال فترة الدراسة.",
      "انخفضت درجات القلق على المقياس الموحد بمعدل 6.3 نقطة بعد التدخل، وهو انخفاض ملحوظ عن خط الأساس.",
    ],
  },
];
