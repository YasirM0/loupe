// A larger, deliberately adversarial test corpus — 4 reference "papers" and
// 12 claims against them, engineered with the same trap categories the
// historical full-roster stress test in run.mjs's comments describes
// (numeric mismatches, fabrications wearing real authors' names,
// cherry-picked partial truths, wrong-source citations, misleading
// vocabulary overlap) but run through the *actual* production pipeline —
// chunking, BM25+embedding retrieval over the whole corpus, NLI
// classification, guards — rather than isolated claim/evidence pairs like
// NLI_TESTSET/RETRIEVAL_TESTSET. English and Indonesian are faithful
// parallel translations (same claims, same expected verdicts, same traps)
// so results are directly comparable across languages, not just within one.

export const EN_SOURCES = [
  {
    name: 'urban-mobility-2023.txt',
    text: `Eliasson, T. (2023). Congestion Pricing and Urban Traffic Patterns in Stockholm. Journal of Transport Policy, 41(2), 88-104.

This study examines the effects of congestion pricing introduced in central Stockholm in 2022. Traffic volume in the priced zone declined by 20 percent within the first year of implementation, based on continuous sensor data from 45 monitoring stations. The reduction was most pronounced during peak morning hours, when volume fell by nearly 28 percent. Survey data collected from residents showed that support for the pricing scheme was strong among downtown residents, with 68 percent expressing approval. However, suburban commuters who relied on the priced corridors for their daily commute expressed considerably less enthusiasm, with only 31 percent reporting a favorable view of the policy. The study did not examine effects on public transit ridership or autonomous vehicle adoption, which fall outside its scope.`,
  },
  {
    name: 'remote-work-productivity.txt',
    text: `Bloom, N. (2022). Remote Work Arrangements and Employee Productivity Outcomes. Journal of Labor Economics, 30(4), 512-540.

This research analyzed productivity self-assessments from 2,400 employees across engineering, product, and sales departments following the shift to remote work. On average, employees reported a 15 percent increase in self-assessed productivity compared to their pre-remote baseline. This effect was strongest in engineering and product roles, where reduced office distractions were cited as the primary driver. Sales teams, in contrast, reported a modest productivity decline, attributing it to reduced face-to-face client contact and slower deal closure. Contrary to concerns raised in earlier commentary, remote employees in this sample reported lower rates of burnout and isolation than their office-based counterparts, with wellbeing scores improving by 9 percent on average.`,
  },
  {
    name: 'vaccine-efficacy-trial.txt',
    text: `Chen, L. (2021). Phase III Efficacy and Safety Results for the XR-19 Vaccine Candidate. New England Journal of Clinical Trials, 12(1), 33-52.

In this randomized, placebo-controlled phase III trial involving 18,300 participants, the vaccine demonstrated 89 percent efficacy against severe illness. Efficacy against symptomatic infection of any severity was somewhat lower, at 71 percent. Safety monitoring found that adverse events were generally mild, with fewer than 4 percent of participants reporting any side effect beyond mild injection-site soreness. No severe adverse events were attributed to the vaccine during the trial period.`,
  },
  {
    name: 'education-technology-2022.txt',
    text: `Alvarez, R. (2022). Tablet-Based Curricula and Reading Comprehension in Elementary Classrooms. Educational Research Quarterly, 27(3), 201-219.

This study evaluated a tablet-based reading curriculum implemented across 40 elementary school classrooms over one academic year. Students using the tablet-based curriculum showed a statistically significant improvement in standardized reading comprehension scores compared to the control group using traditional print materials, with an average gain of 12 percentile points. The improvement was consistent across grade levels three through five. The study did not track cognitive outcomes beyond the one-year trial period and makes no claims about long-term effects.`,
  },
];

// `trap` documents which adversarial category each claim tests — purely
// informational for the report, not used by the scoring logic. `special`
// marks claim 9, whose "correct" pipeline behavior is genuinely ambiguous
// (see stress-run.mjs) rather than a clean pass/fail.
export const EN_CLAIMS = [
  { claim: 'Congestion pricing in Stockholm reduced traffic volume by 20% within the first year (Eliasson, 2023).', expected: 'SUPPORTED', trap: 'direct match' },
  { claim: 'Employees working remotely reported a 15% increase in self-assessed productivity compared to their pre-remote baseline (Bloom, 2022).', expected: 'SUPPORTED', trap: 'paraphrase' },
  { claim: 'The vaccine demonstrated 95% efficacy against severe illness in the phase III trial (Chen, 2021).', expected: 'CONTRADICTED', trap: 'numeric mismatch (89% -> 95%)' },
  { claim: 'The study found no significant improvement in reading scores among students using the tablet-based curriculum (Alvarez, 2022).', expected: 'CONTRADICTED', trap: 'explicit negation of a stated finding' },
  { claim: 'Remote work improved productivity across all departments (Bloom, 2022).', expected: 'PARTIAL', trap: 'cherry-picked overgeneralization (sales declined)' },
  { claim: 'Congestion pricing was universally popular among city residents (Eliasson, 2023).', expected: 'PARTIAL', trap: 'cherry-picked overgeneralization (suburban 31% approval)' },
  { claim: 'Eliasson (2023) found that congestion pricing increased public transit ridership by 40%.', expected: 'UNSUPPORTED', trap: 'fabrication wearing a real, correctly-cited author name' },
  { claim: 'Chen (2021) reported that the vaccine caused severe side effects in 30% of participants.', expected: 'CONTRADICTED', trap: 'fabrication contradicted by explicit safety data' },
  { claim: 'Alvarez (2022) found that congestion pricing reduced downtown traffic by 20%.', expected: 'SPECIAL', trap: 'wrong-source citation — true finding, real author, but attributed to the wrong paper', special: true },
  { claim: 'The study reported that autonomous vehicle adoption increased by 50% during the trial period (Eliasson, 2023).', expected: 'UNSUPPORTED', trap: 'absence — explicitly out of scope' },
  { claim: 'Long-term studies confirm no lasting cognitive benefits from early tablet exposure (Alvarez, 2022).', expected: 'UNSUPPORTED', trap: 'absence on an adjacent, superficially similar topic' },
  { claim: 'Remote work has been linked to increased employee isolation and burnout (Bloom, 2022).', expected: 'CONTRADICTED', trap: 'misleading vocabulary overlap, opposite meaning' },
];

export const ID_SOURCES = [
  {
    name: 'urban-mobility-2023.txt',
    text: `Eliasson, T. (2023). Tarif Kemacetan dan Pola Lalu Lintas Perkotaan di Stockholm. Journal of Transport Policy, 41(2), 88-104.

Studi ini meneliti dampak tarif kemacetan yang diberlakukan di pusat kota Stockholm pada tahun 2022. Volume lalu lintas di zona bertarif menurun sebesar 20 persen dalam tahun pertama penerapan, berdasarkan data sensor berkelanjutan dari 45 stasiun pemantauan. Penurunan paling signifikan terjadi pada jam sibuk pagi, ketika volume turun hampir 28 persen. Data survei yang dikumpulkan dari warga menunjukkan bahwa dukungan terhadap skema tarif ini kuat di kalangan warga pusat kota, dengan 68 persen menyatakan persetujuan. Namun, komuter pinggiran kota yang bergantung pada koridor bertarif untuk perjalanan harian mereka menunjukkan antusiasme yang jauh lebih rendah, dengan hanya 31 persen yang menyatakan pandangan positif terhadap kebijakan tersebut. Studi ini tidak meneliti dampak terhadap jumlah penumpang transportasi umum atau adopsi kendaraan otonom, yang berada di luar cakupan penelitian ini.`,
  },
  {
    name: 'remote-work-productivity.txt',
    text: `Bloom, N. (2022). Pengaturan Kerja Jarak Jauh dan Hasil Produktivitas Karyawan. Journal of Labor Economics, 30(4), 512-540.

Penelitian ini menganalisis penilaian mandiri terhadap produktivitas dari 2.400 karyawan di departemen teknik, produk, dan penjualan setelah peralihan ke kerja jarak jauh. Rata-rata, karyawan melaporkan peningkatan produktivitas mandiri sebesar 15 persen dibandingkan dengan kondisi awal sebelum kerja jarak jauh. Efek ini paling kuat pada peran teknik dan produk, di mana berkurangnya gangguan di kantor disebut sebagai penyebab utama. Sebaliknya, tim penjualan melaporkan penurunan produktivitas yang moderat, yang mereka kaitkan dengan berkurangnya kontak tatap muka dengan klien dan proses penutupan transaksi yang lebih lambat. Bertentangan dengan kekhawatiran yang diangkat dalam komentar sebelumnya, karyawan jarak jauh dalam sampel ini melaporkan tingkat kelelahan dan isolasi yang lebih rendah dibandingkan rekan mereka yang bekerja di kantor, dengan skor kesejahteraan meningkat rata-rata 9 persen.`,
  },
  {
    name: 'vaccine-efficacy-trial.txt',
    text: `Chen, L. (2021). Hasil Efikasi dan Keamanan Fase III untuk Kandidat Vaksin XR-19. New England Journal of Clinical Trials, 12(1), 33-52.

Dalam uji coba fase III acak dan terkontrol plasebo yang melibatkan 18.300 peserta ini, vaksin menunjukkan efikasi sebesar 89 persen terhadap penyakit parah. Efikasi terhadap infeksi bergejala dalam tingkat keparahan apa pun sedikit lebih rendah, yaitu 71 persen. Pemantauan keamanan menemukan bahwa efek samping umumnya ringan, dengan kurang dari 4 persen peserta melaporkan efek samping apa pun selain nyeri ringan di lokasi suntikan. Tidak ada efek samping serius yang dikaitkan dengan vaksin selama masa uji coba.`,
  },
  {
    name: 'education-technology-2022.txt',
    text: `Alvarez, R. (2022). Kurikulum Berbasis Tablet dan Pemahaman Membaca di Kelas Sekolah Dasar. Educational Research Quarterly, 27(3), 201-219.

Studi ini mengevaluasi kurikulum membaca berbasis tablet yang diterapkan di 40 kelas sekolah dasar selama satu tahun ajaran. Siswa yang menggunakan kurikulum berbasis tablet menunjukkan peningkatan yang signifikan secara statistik dalam skor pemahaman membaca standar dibandingkan dengan kelompok kontrol yang menggunakan materi cetak tradisional, dengan rata-rata peningkatan 12 poin persentil. Peningkatan ini konsisten di seluruh tingkat kelas tiga hingga lima. Studi ini tidak melacak hasil kognitif setelah masa uji coba satu tahun dan tidak membuat klaim apa pun tentang efek jangka panjang.`,
  },
];

export const ID_CLAIMS = [
  { claim: 'Tarif kemacetan di Stockholm mengurangi volume lalu lintas sebesar 20% dalam tahun pertama (Eliasson, 2023).', expected: 'SUPPORTED', trap: 'direct match' },
  { claim: 'Karyawan yang bekerja jarak jauh melaporkan peningkatan produktivitas mandiri sebesar 15% dibandingkan kondisi awal sebelum kerja jarak jauh (Bloom, 2022).', expected: 'SUPPORTED', trap: 'paraphrase' },
  { claim: 'Vaksin menunjukkan efikasi 95% terhadap penyakit parah dalam uji coba fase III (Chen, 2021).', expected: 'CONTRADICTED', trap: 'numeric mismatch (89% -> 95%)' },
  { claim: 'Studi tersebut tidak menemukan peningkatan signifikan dalam skor membaca di antara siswa yang menggunakan kurikulum berbasis tablet (Alvarez, 2022).', expected: 'CONTRADICTED', trap: 'explicit negation of a stated finding' },
  { claim: 'Kerja jarak jauh meningkatkan produktivitas di semua departemen (Bloom, 2022).', expected: 'PARTIAL', trap: 'cherry-picked overgeneralization (sales declined)' },
  { claim: 'Tarif kemacetan populer secara universal di kalangan warga kota (Eliasson, 2023).', expected: 'PARTIAL', trap: 'cherry-picked overgeneralization (suburban 31% approval)' },
  { claim: 'Eliasson (2023) menemukan bahwa tarif kemacetan meningkatkan jumlah penumpang transportasi umum sebesar 40%.', expected: 'UNSUPPORTED', trap: 'fabrication wearing a real, correctly-cited author name' },
  { claim: 'Chen (2021) melaporkan bahwa vaksin menyebabkan efek samping serius pada 30% peserta.', expected: 'CONTRADICTED', trap: 'fabrication contradicted by explicit safety data' },
  { claim: 'Alvarez (2022) menemukan bahwa tarif kemacetan mengurangi lalu lintas pusat kota sebesar 20%.', expected: 'SPECIAL', trap: 'wrong-source citation — true finding, real author, but attributed to the wrong paper', special: true },
  { claim: 'Studi tersebut melaporkan bahwa adopsi kendaraan otonom meningkat sebesar 50% selama masa uji coba (Eliasson, 2023).', expected: 'UNSUPPORTED', trap: 'absence — explicitly out of scope' },
  { claim: 'Studi jangka panjang mengonfirmasi tidak ada manfaat kognitif yang bertahan dari paparan tablet sejak dini (Alvarez, 2022).', expected: 'UNSUPPORTED', trap: 'absence on an adjacent, superficially similar topic' },
  { claim: 'Kerja jarak jauh dikaitkan dengan peningkatan isolasi dan kelelahan karyawan (Bloom, 2022).', expected: 'CONTRADICTED', trap: 'misleading vocabulary overlap, opposite meaning' },
];
