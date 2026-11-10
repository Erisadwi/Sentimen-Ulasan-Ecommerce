
import pandas as pd
import os
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier


# ============================================================
# 1. KONFIGURASI
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = (
    BASE_DIR.parent
    / 'dataset'
    / 'ulasan_shopee_oktober_2026_clean.csv'
)

OUTPUT_DIR = BASE_DIR.parent / 'hasil_model'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20


print("=" * 60)
print("       IMPLEMENTASI MODEL MACHINE LEARNING")
print("          ANALISIS SENTIMEN SHOPEE")
print("=" * 60)


# ============================================================
# 2. MEMUAT DATASET
# ============================================================

print("\n[1] MEMUAT DATASET")

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset tidak ditemukan: {DATA_PATH}"
    )

df = pd.read_csv(
    DATA_PATH,
    encoding='utf-8-sig'
)

# Memastikan kolom tersedia
kolom_wajib = ['ulasan_clean', 'sentimen']

for kolom in kolom_wajib:
    if kolom not in df.columns:
        raise ValueError(
            f"Kolom {kolom} tidak ditemukan."
        )

# Menghapus data kosong
df = df.dropna(
    subset=['ulasan_clean', 'sentimen']
).copy()

df['ulasan_clean'] = (
    df['ulasan_clean'].astype(str).str.strip()
)

df['sentimen'] = (
    df['sentimen'].astype(str).str.strip()
)

df = df[df['ulasan_clean'] != ''].copy()

# Memastikan label biner
df = df[
    df['sentimen'].isin(['Positif', 'Negatif'])
].copy()

if df.empty:
    raise ValueError("Dataset tidak memiliki data valid.")

print("Dataset berhasil dimuat.")
print("Jumlah data:", len(df))

print("\nDistribusi sentimen:")
print(df['sentimen'].value_counts())


# ============================================================
# 3. MENENTUKAN FITUR DAN LABEL
# ============================================================

print("\n[2] MENENTUKAN FITUR DAN LABEL")

X = df['ulasan_clean']
y = df['sentimen']

print("Fitur : ulasan_clean")
print("Label : sentimen")


# ============================================================
# 4. PEMBAGIAN DATA TRAINING DAN TESTING
# ============================================================

print("\n[3] PEMBAGIAN DATA")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print("Jumlah data training:", len(X_train))
print("Jumlah data testing :", len(X_test))


# ============================================================
# 5. EKSTRAKSI FITUR TF-IDF
# ============================================================

print("\n[4] EKSTRAKSI FITUR TF-IDF")

tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=1,
    sublinear_tf=True
)

# Pembelajaran TF-IDF dari data training
X_train_tfidf = tfidf.fit_transform(X_train)

# Transformasi data testing
X_test_tfidf = tfidf.transform(X_test)

print("Ekstraksi fitur berhasil.")
print("Jumlah fitur:", len(tfidf.get_feature_names_out()))
print("Matriks training:", X_train_tfidf.shape)
print("Matriks testing :", X_test_tfidf.shape)


# ============================================================
# 6. IMPLEMENTASI MODEL NAIVE BAYES
# ============================================================

print("\n[5] IMPLEMENTASI NAIVE BAYES")

model_nb = MultinomialNB(
    alpha=1.0
)

# Proses training
model_nb.fit(
    X_train_tfidf,
    y_train
)

print("Training Naive Bayes selesai.")

# Prediksi data testing
prediksi_nb = model_nb.predict(
    X_test_tfidf
)

print("Prediksi Naive Bayes selesai.")


# ============================================================
# 7. IMPLEMENTASI MODEL K-NEAREST NEIGHBOR
# ============================================================

print("\n[6] IMPLEMENTASI KNN")

model_knn = KNeighborsClassifier(
    n_neighbors=5,
    metric='cosine',
    algorithm='brute',
    weights='distance'
)

# Proses training
model_knn.fit(
    X_train_tfidf,
    y_train
)

print("Training KNN selesai.")

# Prediksi data testing
prediksi_knn = model_knn.predict(
    X_test_tfidf
)

print("Prediksi KNN selesai.")


# ============================================================
# 8. MENYIMPAN HASIL PREDIKSI AWAL
# ============================================================

print("\n[7] MENYIMPAN HASIL PREDIKSI")

df_prediksi = pd.DataFrame({
    'ulasan_clean': X_test.values,
    'sentimen_aktual': y_test.values,
    'prediksi_naive_bayes': prediksi_nb,
    'prediksi_knn': prediksi_knn
})

df_prediksi.to_csv(
    OUTPUT_DIR / 'hasil_prediksi.csv',
    index=False,
    encoding='utf-8-sig'
)

print("Hasil prediksi berhasil disimpan.")


# ============================================================
# 9. MENYIMPAN MODEL DAN TF-IDF
# ============================================================

print("\n[8] MENYIMPAN MODEL")

joblib.dump(
    tfidf,
    OUTPUT_DIR / 'tfidf_vectorizer.pkl'
)

joblib.dump(
    model_nb,
    OUTPUT_DIR / 'naive_bayes.pkl'
)

joblib.dump(
    model_knn,
    OUTPUT_DIR / 'knn.pkl'
)

print("TF-IDF berhasil disimpan.")
print("Naive Bayes berhasil disimpan.")
print("KNN berhasil disimpan.")


# ============================================================
# 10. RINGKASAN PROSES
# ============================================================

print("\n" + "=" * 60)
print("       IMPLEMENTASI MODEL SELESAI")
print("=" * 60)

print("Jumlah dataset :", len(df))
print("Data training  :", len(X_train))
print("Data testing   :", len(X_test))

print("\nModel yang berhasil dibuat:")
print("1. Multinomial Naive Bayes")
print("2. K-Nearest Neighbor")

print("\nLokasi penyimpanan:")
print(OUTPUT_DIR)

print("\nSeluruh pipeline berhasil dijalankan.")
print("=" * 60)
