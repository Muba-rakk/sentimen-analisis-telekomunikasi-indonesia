# Progress Report — Sentiment Analysis Provider Telekomunikasi Indonesia

## Ringkasan Proyek

Proyek ini melakukan analisis sentimen terhadap tiga provider telekomunikasi Indonesia (Telkomsel, Indosat, XL Axiata) menggunakan data ulasan Google Play Store. Metode klasifikasi: **Logistic Regression** dengan fitur **TF-IDF + Lexicon-Based Features**, mengikuti metodologi **CRISP-DM**.

---

## Progress CRISP-DM

### Fase 1 & 2: Business & Data Understanding → `notebooks/01_eda.ipynb`
**Status: ✅ Selesai**

Notebook ini melakukan:
1. **Load Data** — Memanggil `load_all()` dari `src.data_loader` yang membaca 2 file CSV raw (Play Store & Twitter), lalu memfilter CS reply, spam, duplikat
2. **Visualisasi Distribusi** — Bar chart jumlah data per provider dan per sumber, histogram panjang tweet, heatmap missing values
3. **Inspeksi Sampel** — Menampilkan 5 contoh tweet per provider untuk memahami pola bahasa

**Hasil:** 4.842 tweet setelah filtering awal (sebelum preprocessing & labeling)

### Fase 3: Data Preparation → `notebooks/02_data_preparation.ipynb`
**Status: ✅ Selesai (Diperbarui 26 Mei 2026)**

Notebook ini melakukan:
1. **Load & filter data** (sama seperti notebook 01)
2. **Preprocessing** — Panggil `run_preprocessing_pipeline()` dari `src.preprocessor`
   - **Clean text**: lowercase, hapus URL/mention/hashtag/emoji/punctuation
   - **Slang Normalization**: ubah kata gaul ke baku (misal: "ngeleg" → "lambat", "kouta" → "kuota") pakai `SLANG_DICT` dari config
   - **Stopword removal**: hapus kata umum Bahasa Indonesia + kata ganti ("aku", "kamu", "sy") + slang Twitter
   - **Stemming**: potong imbuhan ke kata dasar pakai PySastrawi
3. **Labeling** — Panggil `run_labeling()` dari `src.labeler`
   - Hitung jumlah kata positif & negatif per tweet dari kamus lexicon yang sudah disterilkan dari kata-kata saling tumpang-tindih.
   - Label: 1 (positif) jika pos_count > neg_count, selainnya 0 (negatif)
   - Drop baris dengan pos_count=0 dan neg_count=0 (tidak terlabeli)
4. **Simpan hasil** ke `data/processed/data_labeled.csv`

**Hasil Baru:** 4.347 tweet berlabel (2912 negatif, 1435 positif). Bentuk kata jauh lebih baku karena ada translasi bahasa gaul sebelum stemming.
- Indosat: 1326 (35.14% positif)
- Telkomsel: 1400 (34.93% positif)
- XL: 1621 (29.61% positif)

### Fase 4: Modeling → `notebooks/03_modeling.ipynb`
**Status: ⏳ Belum dijalankan**

### Fase 5: Evaluation → `notebooks/04_evaluation.ipynb`
**Status: ⏳ Belum dijalankan**

### Fase 6: Deployment → `notebooks/05_deployment.ipynb`
**Status: ⏳ Belum dijalankan**

---

## Penjelasan Teknis Source Code (`src/`)

### `src/config.py` — Konfigurasi Pusat

| Baris | Variabel | Fungsi |
|-------|----------|--------|
| 3 | `ROOT_DIR` | Path absolut root proyek (parent dari `src/`) |
| 4 | `DATA_RAW` | Path ke `data/raw/` — tempat CSV hasil scraping |
| 5 | `DATA_PROC` | Path ke `data/processed/` — output preprocessing |
| 6 | `LEXICON` | Path ke `data/lexicon/` — kamus sentimen |
| 7-8 | `RESULTS`, `FIGURES` | Path hasil model dan gambar |
| 10 | `RANDOM_STATE` | Seed 42 untuk reprodusibilitas |
| 11 | `TEST_SIZE` | 30% data untuk testing |
| 12 | `LR_PARAMS` | Default parameter Logistic Regression: solver `lbfgs`, max_iter 1000 |
| 13 | `TUNING_C` | Kandidat nilai C (regularisasi) untuk GridSearch: 0.01, 0.1, 1, 10, 100 |
| 15 | `PROVIDERS` | List 3 provider: indosat, telkomsel, xl |
| 18-22 | `TWITTER_STOPWORDS` | Stopword slang Twitter: "yg", "dg", "bgt", "gw", "loe", dll |

**Logic:** File ini adalah satu-satunya tempat konfigurasi. Semua module `src/` lain import dari sini. Dengan `pathlib.Path`, path otomatis menyesuaikan sistem operasi.

---

### `src/data_loader.py` — Loading & Filtering Data

**`load_new_data()` (baris 5-31):**
- Membaca `data_playstore.csv` dan `data_twitter.csv` dari `DATA_RAW`
- Jika file tidak ada → print warning + skip
- Jika file kosong → tangkap exception `EmptyDataError`
- Gabung semua file jadi satu DataFrame dengan `pd.concat()`
- **Baris 25:** `fillna('')` — isi NaN dengan string kosong agar operasi string tidak error
- **Baris 27:** Filter baris dengan `tweet_raw` kosong/strip habis

**`filter_cs_replies()` (baris 33-42):**
- **Baris 38:** `str.contains('indosat|telkomsel|xlaxiata')` — pola regex untuk username akun resmi provider
- **Baris 41:** `df[~cs_mask]` — tanda `~` membalik boolean mask, jadi hanya baris BUKAN CS yang dipertahankan
- **Baris 40:** Print jumlah CS yang di-drop untuk transparansi

**`filter_spam()` (baris 44-56):**
- **Baris 46-51:** Array pola regex spam:
  - `\bwtb\b|\bwts\b` — "want to buy/want to sell" (jual-beli)
  - `\bjasa cv\b` — jasa pembuatan CV
  - `\b(the|and|is|to)\b` — kata umum Bahasa Inggris (indikasi non-Indonesia)
  - `\b(trknet|türknet|türksat)\b` — provider Turki
- **Baris 52:** `'|'.join()` — gabung semua pola dengan OR
- **Baris 53:** `case=False` — case insensitive, `na=False` — NaN dianggap tidak match
- **Baris 56:** Filter dengan negasi `~mask` — hanya tweet TIDAK mengandung pola spam

**`filter_telecom_only()` (baris 58-71):**
- **Baris 61-65:** Pola regex keyword telekomunikasi:
  - Nama provider: `\bindosat\b`, `\btelkomsel\b`, `\bxl\b`, dll.
  - Istilah teknis: `\bsinyal\b`, `\bkuota\b`, `\binternet\b`, dll.
  - `\b` adalah *word boundary* — memastikan kata utuh, bukan substring
- **Baris 68-71:** Hanya tweet yang mengandung minimal 1 keyword dipertahankan

**`load_all()` (baris 73-100):**
- **Pipeline urutan:** Load → Filter CS → Filter Spam → Filter Telecom (khusus socmed) → Drop Duplikat
- **Baris 89-94:** Filter telecom hanya untuk Twitter/Facebook (sumber yang mungkin berisi konten non-telekomunikasi). Play Store review sudah pasti tentang app provider.
- **Baris 97:** `drop_duplicates(subset=['tweet_raw'])` — hapus tweet duplikat berdasarkan teksnya

---

### `src/preprocessor.py` — Text Preprocessing Pipeline

**Inisialisasi (baris 10-20):**
- **Baris 10-11:** Buat stemmer Sastrawi sekali di module level (singleton) agar tidak dibuat ulang setiap preprocessing
- **Baris 15-17:** Cek apakah NLTK stopwords sudah di-download, jika belum download otomatis (quiet=True agar tidak spam output)
- **Baris 19-20:** Gabung stopwords NLTK Bahasa Indonesia + custom Twitter stopwords

**`clean_text()` (baris 22-34):**
- **Baris 23:** `str(text).lower()` — lowercase
- **Baris 25:** `re.sub(r'http\S+|www\S+')` — regex hapus URL (http://, https://, www.)
- **Baris 27:** `re.sub(r'@\w+')` — hapus mention username
- **Baris 29:** `re.sub(r'#\w+')` — hapus hashtag
- **Baris 31:** `re.sub(r'[^a-z\s]', ' ')` — **inti:** hapus semua karakter yang bukan huruf a-z dan spasi. Ini menghilangkan: emoji, angka, tanda baca, karakter asing
- **Baris 33:** `re.sub(r'\s+', ' ')` — ganti spasi ganda/baris baru/tab jadi 1 spasi + strip

**`remove_stopwords()` (baris 36-39):**
- `text.split()` → tokenisasi (pisah per spasi)
- List comprehension: jaga token yang TIDAK ada di `stop_words`
- Gabung kembali dengan spasi

**`apply_stemming()` (baris 41-42):**
- Delegasi ke `stemmer.stem()` dari PySastrawi
- Contoh: "berjalan" → "jalan", "membangunkan" → "bangun"

**`run_preprocessing_pipeline()` (baris 44-60):**
- **Baris 45:** `tqdm.pandas()` — aktivasi progress bar untuk pandas
- **Baris 47-51:** Inner function `process_row` yang menjalankan 3 tahap berurutan: clean → stopword → stem
- **Baris 53:** `progress_apply()` — apply dengan progress bar
- **Baris 57:** Filter baris yang setelah preprocessing jadi string kosong (misal tweet hanya berisi emoji)

---

### `src/labeler.py` — Lexicon-Based Auto Labeling

**`load_lexicon()` (baris 4-19):**
- Baca `positive_words.txt` dan `negative_words.txt` dari folder `LEXICON`
- **Baris 8:** `line.strip().split('\t')[0]` — format file: kata di kolom 1 tab-separated
- **Baris 9:** Skip baris header (word) dan kata kosong
- Return dalam bentuk `set` (hash set) agar pencarian O(1) — cepat

**`count_sentiment_words()` (baris 21-25):**
- **Baris 22:** Tokenisasi teks
- **Baris 23:** `sum(1 for word in words if word in pos_words)` — hitung kata yang ada di set positif. Operasi `in` di set = O(1)
- **Baris 24:** Sama untuk negatif
- Return tuple (pos_count, neg_count)

**`assign_label()` (baris 27-28):**
- Logika sederhana: jika pos_count > neg_count → 1 (positif), selainnya 0 (negatif)
- **Kelemahan:** Jika pos_count == neg_count, dianggap negatif (0). Ini bisa diperbaiki dengan threshold atau tie-breaking.

**`run_labeling()` (baris 30-54):**
- **Baris 31:** Load lexicon sekali
- **Baris 33-39:** Iterasi setiap teks, hitung pos/neg count, kumpulkan di list
- **Baris 42-43:** Tambahkan kolom `pos_count` dan `neg_count` ke DataFrame
- **Baris 47:** `df[(df['pos_count'] > 0) | (df['neg_count'] > 0)]` — hanya baris yang punya setidaknya 1 kata sentimen
- **Baris 52:** `apply(lambda row: assign_label(...), axis=1)` — terapkan fungsi per baris (axis=1 = horizontal)

---

### `src/features.py` — Feature Engineering

**`build_tfidf()` (baris 9-16):**
- **Baris 13:** `TfidfVectorizer()` — tanpa parameter khusus, menggunakan default: max_features=None (semua kata), analyzer='word', ngram_range=(1,1)
- **Baris 14:** `fit_transform()` pada TRAIN — belajar vocabulary + transform simultan
- **Baris 15:** `transform()` pada TEST — transformasi saja, tanpa belajar ulang (mencegah data leakage)

**`build_lexicon_features()` (baris 19-29):**
- **Baris 25:** `df.loc[train_idx][['pos_count', 'neg_count']].values` — ambil 2 kolom angka sebagai array numpy
- **Baris 26:** `MinMaxScaler()` — scaling ke rentang [0,1]
- **Baris 27:** Scaler di-fit pada TRAIN saja
- **Baris 28:** Transform TEST dengan scaler yang sudah di-fit (cek: tidak boleh fit_transform ulang)

**`combine_features()` (baris 32-36):**
- **Baris 36:** `scipy_hstack([X_tfidf, csr_matrix(lex_features)])` — gabung matriks horizontal
  - `csr_matrix(lex_features)` — konversi array dense ke sparse (efisien memory)
  - `format='csr'` — Compressed Sparse Row format
  - TF-IDF asli sparse + fitur lexicon jadi 1 matriks besar

---

### `src/model.py` — Logistic Regression

**`train()` (baris 6-9):**
- **Baris 7:** `LogisticRegression(**params)` — unpack dict parameter. Contoh: `LR_PARAMS = {"solver": "lbfgs"}` jadi `LogisticRegression(solver='lbfgs')`
- **Baris 8:** `model.fit(X_train, y_train)` — training Logistic Regression

**`tune()` (baris 11-21):**
- **Baris 12:** `param_grid = {'C': C_values}` — C adalah inverse regularization strength. C kecil → regularisasi kuat (hindari overfitting). C besar → regularisasi lemah
- **Baris 13-19:** `GridSearchCV`:
  - `cv=5` — 5-fold cross-validation
  - `scoring='accuracy'` — optimasi akurasi
  - `n_jobs=-1` — gunakan semua CPU core
- **Baris 21:** Return `best_estimator_` (model terbaik), `best_params_['C']` (C optimal), `best_score_` (rata-rata CV score)

**`predict()` (baris 23-24):**
- Delegasi ke `model.predict()` — return array 0/1

---

### `src/evaluator.py` — Evaluasi & Visualisasi

**`compute_metrics()` (baris 12-18):**
- **Baris 14:** `accuracy_score()` — (TP+TN)/(TP+TN+FP+FN)
- **Baris 15-17:** `average="weighted"` — hitung metrik per kelas lalu rata-rata tertimbang (weighted by support). `zero_division=0` — jika ada kelas tanpa prediksi, precision dianggap 0 (bukan error)
- Return dict 4 metrik

**`plot_confusion_matrix()` (baris 21-31):**
- **Baris 22:** `confusion_matrix()` — matrix 2x2: [ [TN, FP], [FN, TP] ]
- **Baris 24:** `sns.heatmap(annot=True, fmt="d")` — tampilkan angka sebagai integer
- **Baris 30:** `savefig()` — simpan ke file (di `results/figures/`)
- **Baris 31:** `plt.close()` — tutup figure untuk hemat memory

**`plot_comparison()` (baris 34-58):**
- **Baris 41:** `np.arange(len(metrics))` — array [0, 1, 2, 3] untuk posisi x
- **Baris 42:** `width = 0.35` — lebar bar (dari total space 0.7 yang terisi)
- **Baris 45-46:** Dua bar chart berdampingan: offset -width/2 dan +width/2
- **Baris 52:** `set_ylim(0, 1.0)` — skala y dari 0 ke 1 (karena semua metrik dalam rentang 0-1)

**`evaluate_per_provider()` (baris 61-75):**
- **Baris 63:** Iterasi per provider unik
- **Baris 65:** Transform teks dengan `vectorizer.transform()` (sudah di-fit sebelumnya)
- **Baris 66-70:** Jika menggunakan lexicon features: ambil pos_count/neg_count, scale, hstack dengan TF-IDF
- **Baris 72-73:** Prediksi + hitung akurasi per provider
- **Baris 75:** Return DataFrame: [provider, accuracy]

**`save_results()` (baris 78-81):**
- Simpan dict hasil evaluasi ke CSV

---

## Glosarium / Kamus Istilah

### Machine Learning & Matematika
| Istilah | Arti |
|---------|------|
| **Logistic Regression** | Model klasifikasi biner yang memprediksi probabilitas kelas menggunakan fungsi sigmoid |
| **TF-IDF** | Term Frequency-Inverse Document Frequency — teknik memberi bobot pada kata berdasarkan frekuensi di dokumen dan inverse frekuensi di seluruh korpus |
| **C (Regularization)** | Parameter yang mengontrol kekuatan regularisasi. Kecil = regularisasi kuat (simpler model). Besar = regularisasi lemah (lebih fleksibel) |
| **GridSearchCV** | Teknik mencari hyperparameter optimal dengan mencoba semua kombinasi + cross-validation |
| **Cross-validation (CV)** | Membagi data jadi k fold, latih di k-1 fold, evaluasi di 1 fold sisanya. Ulang k kali |
| **MinMaxScaler** | Normalisasi fitur ke rentang [0,1] dengan rumus: (x - min)/(max - min) |
| **hstack** | Horizontal stack — menggabung matriks secara kolom (menambah fitur) |
| **csr_matrix** | Compressed Sparse Row — format penyimpanan matriks sparse (banyak nilai 0) yang hemat memory |
| **Accuracy** | (TP + TN) / Total — proporsi prediksi benar |
| **Precision** | TP / (TP + FP) — dari semua yang diprediksi positif, berapa yang benar positif |
| **Recall** | TP / (TP + FN) — dari semua yang sebenarnya positif, berapa yang berhasil dideteksi |
| **F1-Score** | Harmonic mean precision & recall = 2 * (P * R) / (P + R) |
| **Weighted Average** | Rata-rata metrik per kelas yang ditimbang oleh jumlah sampel di tiap kelas |
| **Confusion Matrix** | Tabel 2x2 atau lebih yang menunjukkan True Positive, True Negative, False Positive, False Negative |
| **ReLU** | Rectified Linear Unit — fungsi aktivasi f(x) = max(0, x) di neural network (tidak dipakai di proyek ini) |
| **Sigmoid** | Fungsi aktivasi berbentuk S f(x) = 1/(1+e^-x). Logistic Regression menggunakannya untuk output probabilitas |

### Natural Language Processing (NLP)
| Istilah | Arti |
|---------|------|
| **Stopwords** | Kata umum yang tidak memiliki makna signifikan (dan, atau, yang, di, ke) yang dihapus saat preprocessing |
| **Stemming** | Proses memotong imbuhan kata menjadi kata dasar (berjalan → jalan, memakan → makan) |
| **Tokenisasi** | Memecah kalimat menjadi unit-unit kecil (token/kata) |
| **Lexicon** | Kamus kata-kata yang sudah dikategorikan (positif/negatif) untuk analisis sentimen |
| **Word Boundary `\b`** | Anchor regex yang memastikan kecocokan kata utuh, bukan substring (misal `\bxl\b` tidak cocok dengan "example") |
| **Vectorizer** | Alat untuk mengubah teks menjadi vektor numerik yang bisa diproses ML |
| **Fitur (Feature)** | Variabel input yang digunakan model untuk membuat prediksi |

### Python & Library
| Istilah | Arti |
|---------|------|
| **`pathlib.Path`** | Modul Python untuk manipulasi path file secara OOP dan cross-platform |
| **`pd.read_csv(lineterminator='\n')`** | Membaca CSV dengan pemisah baris eksplisit, mencegah error jika ada newline di dalam teks |
| **`pd.concat()`** | Menggabung dua DataFrame secara vertikal (baris) |
| **`str.contains(pattern, na=False)`** | Mencari baris yang mengandung pola regex. `na=False` → NaN dianggap tidak match |
| **`drop_duplicates(subset=[...])`** | Menghapus baris duplikat berdasarkan kolom tertentu |
| **`tqdm.pandas()`** | Mengaktifkan progress bar untuk operasi pandas seperti `progress_apply()` |
| **`sns.heatmap(annot=True)`** | Membuat heatmap dengan angka yang ditampilkan di setiap sel |
| **`matplotlib.use('Agg')`** | Backend non-interaktif untuk menyimpan gambar tanpa perlu display |
| **`scipy.sparse.hstack`** | Menggabung matriks sparse horizontal |
| **PySastrawi** | Library stemming Bahasa Indonesia yang mengimplementasi algoritma Nazief-Adriani |
| **NLTK** | Natural Language Toolkit — toolkit NLP Python, digunakan untuk stopwords |

### CRISP-DM & Proyek
| Istilah | Arti |
|---------|------|
| **CRISP-DM** | Cross-Industry Standard Process for Data Mining — metodologi standar industri untuk proyek data mining |
| **EDA** | Exploratory Data Analysis — eksplorasi awal data untuk memahami pola, distribusi, anomali |
| **Data Leakage** | Kebocoran informasi dari test set ke training set (kesalahan fatal: model terlihat pintar tapi gagal di data baru) |
| **Reprodusibilitas** | Kemampuan untuk mendapatkan hasil yang sama persis jika dijalankan ulang (dengan `random_state=42`) |
| **Feature Engineering** | Proses membuat fitur baru dari data mentah untuk meningkatkan performa model |
| **Baseline Model** | Model sederhana sebagai tolok ukur minimal (Model A = TF-IDF only) |
