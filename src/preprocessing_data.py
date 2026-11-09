import pandas as pd
import time
import os
import re
from google_play_scraper import reviews, Sort

# ==============================================================================
# 1. KONFIGURASI PENGAMBILAN DATA (KHUSUS OKTOBER 2026)
# ==============================================================================

APP_ID = 'com.shopee.id'

# Rentang waktu: 01 Oktober 2026 00:00:00 WIB s.d. 31 Oktober 2026 23:59:59 WIB
START_DATE = pd.Timestamp('2026-10-01', tz='Asia/Jakarta')
END_DATE = pd.Timestamp('2026-11-01', tz='Asia/Jakarta')

all_reviews = []
continuation_token = None

print("==================================================")
print("          1. TAHAP WEB SCRAPING ULASAN SHOPEE     ")
print("==================================================")
print("Tanggal mulai : 01 Oktober 2026")
print("Tanggal akhir : 31 Oktober 2026")
print("Memulai proses scraping dari Google Play Store...\n")


# ==============================================================================
# 2. FUNGSI KONVERSI TANGGAL KE WIB
# ==============================================================================

def convert_to_wib(date):
    date = pd.Timestamp(date)
    if date.tzinfo is None:
        return date.tz_localize('Asia/Jakarta')
    else:
        return date.tz_convert('Asia/Jakarta')


# ==============================================================================
# 3. WEB SCRAPING DATA ULASAN
# ==============================================================================

while True:
    result, continuation_token = reviews(
        APP_ID,
        lang='id',
        country='id',
        sort=Sort.NEWEST,
        count=200,
        continuation_token=continuation_token
    )

    if not result:
        print("Tidak ada data ulasan lagi.")
        break

    # Konversi tanggal ulasan ke WIB
    batch_dates = [convert_to_wib(r['at']) for r in result]
    oldest_date = min(batch_dates)

    # Mengambil ulasan sesuai rentang Oktober 2026
    for r, review_date in zip(result, batch_dates):
        if START_DATE <= review_date < END_DATE:
            all_reviews.append(r)

    print(f"Data ulasan dikumpulkan: {len(all_reviews)} | Ulasan tertua dalam batch: {oldest_date.strftime('%Y-%m-%d %H:%M:%S')}")

    # Berhenti jika sudah melewati tanggal awal (01 Oktober 2026)
    if oldest_date < START_DATE:
        print("\nBatas tanggal awal (01 Oktober 2026) tercapai.")
        break

    if continuation_token is None:
        print("\nSeluruh data ulasan yang tersedia telah diperiksa.")
        break

    time.sleep(1.5)


# ==============================================================================
# 4. MEMBENTUK DAN MENYELEKSI DATASET MENTAH
# ==============================================================================

df = pd.DataFrame(all_reviews)

if df.empty:
    raise ValueError("Tidak ditemukan ulasan pada rentang Oktober 2026. Periksa kembali ketersediaan data historis.")

# Menghapus data duplikat berdasarkan reviewId
df = df.drop_duplicates(subset='reviewId')

# Memilih dan mengubah nama kolom
df = df[['reviewId', 'userName', 'content', 'score', 'at', 'thumbsUpCount']]
df.columns = ['id_ulasan', 'nama_pengguna', 'ulasan', 'rating', 'tanggal', 'jumlah_like']

# Mengubah format tanggal
df['tanggal'] = df['tanggal'].apply(convert_to_wib)

# Filter ketat rentang Oktober 2026
df_filtered = df[(df['tanggal'] >= START_DATE) & (df['tanggal'] < END_DATE)].copy()

# Menghapus ulasan kosong/null
df_filtered = df_filtered.dropna(subset=['ulasan'])
df_filtered = df_filtered[df_filtered['ulasan'].str.strip() != ''].copy()

if df_filtered.empty:
    raise ValueError("Tidak ada ulasan valid setelah seleksi data.")


# ==============================================================================
# 5. PELABELAN SENTIMEN BINER (POSITIF & NEGATIF)
# ==============================================================================

def labeling(rating):
    if rating <= 2:
        return 'Negatif'
    elif rating == 3:
        return 'Netral'
    else:
        return 'Positif'

df_filtered['sentimen'] = df_filtered['rating'].apply(labeling)

# Menghapus kelas 'Netral' untuk fokus klasifikasi Biner (Positif vs Negatif)
df_filtered = df_filtered[df_filtered['sentimen'] != 'Netral'].copy()


# ==============================================================================
# 6. MENYIMPAN DATASET MENTAH HASIL SCRAPING
# ==============================================================================

# Membuat folder 'dataset' sejajar dengan folder 'src' (naik 1 tingkat via '../')
os.makedirs('../dataset', exist_ok=True)

df_filtered = df_filtered.sort_values(by='tanggal')
df_excel = df_filtered.copy()
df_excel['tanggal'] = df_excel['tanggal'].dt.tz_localize(None)

# Simpan CSV dan Excel Mentah
df_excel.to_csv('../dataset/ulasan_shopee_oktober_2026.csv', index=False, encoding='utf-8-sig')
df_excel.to_excel('../dataset/ulasan_shopee_oktober_2026.xlsx', index=False)

print("\n==================================================")
print("            RINGKASAN HASIL SCRAPING              ")
print("==================================================")
print("Total ulasan Oktober 2026 :", len(df_filtered))
print("Tanggal paling awal       :", df_filtered['tanggal'].min().strftime('%Y-%m-%d %H:%M:%S'))
print("Tanggal paling akhir      :", df_filtered['tanggal'].max().strftime('%Y-%m-%d %H:%M:%S'))
print("\nDistribusi Sentimen (Biner):")
print(df_filtered['sentimen'].value_counts())
print("\nDataset mentah tersimpan di folder: ../dataset/\n")


# ==============================================================================
# 7. TAHAP PRAPEMROSESAN DATA TEKS (TEXT PREPROCESSING)
# ==============================================================================

print("==================================================")
print("       2. TAHAP PRAPEMROSESAN DATA TEKS           ")
print("==================================================")

# A. Kamus Normalisasi Kata Slang / Typo (Dapat ditambah sesuai kebutuhan)
kamus_normalisasi = {
    'bgt': 'banget',
    'bgus': 'bagus',
    'sgt': 'sangat',
    'gatau': 'tidak tahu',
    'ga': 'tidak',
    'gak': 'tidak',
    'ngga': 'tidak',
    'nggak': 'tidak',
    'tdk': 'tidak',
    'tpi': 'tapi',
    'sdh': 'sudah',
    'udh': 'sudah',
    'dgn': 'dengan',
    'krn': 'karena',
    'pnp': 'penipuan',
    'cepet': 'cepat',
    'lamaaa': 'lama',
    'sy': 'saya',
    'pake': 'pakai',
    'ongkirr': 'ongkir'
}

# B. Daftar Stopwords (Kata Hubung/Tugas yang Dieliminasi)
stopwords_indonesia = set([
    'yang', 'di', 'ke', 'dari', 'ini', 'itu', 'dan', 'atau', 
    'dengan', 'untuk', 'pada', 'adalah', 'sebagai', 'akan', 
    'saja', 'juga', 'ada', 'bisa', 'saya', 'kami', 'kita'
])

def preprocess_text(text):
    # 1. Text Cleaning (Menghapus URL, angka, simbol, tanda baca)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    
    # 2. Case Folding (Mengubah ke huruf kecil)
    text = text.lower().strip()
    
    # 3. Tokenizing (Memotong teks menjadi list kata)
    tokens = text.split()
    
    # 4. Normalization (Mengganti kata tidak baku/slang)
    tokens = [kamus_normalisasi.get(word, word) for word in tokens]
    
    # 5. Stopwords Removal (Membuang kata hubung yang tidak bernilai sentimen)
    tokens = [word for word in tokens if word not in stopwords_indonesia]
    
    # Menggabungkan kembali menjadi string teks bersih
    return " ".join(tokens)

# Eksekusi prapemrosesan teks
print("Sedang memproses teks ulasan...")
df_filtered['ulasan_clean'] = df_filtered['ulasan'].apply(preprocess_text)

# Menghapus ulasan yang menjadi kosong setelah pembersihan stopwords/noise
df_processed = df_filtered[df_processed_clean := df_filtered['ulasan_clean'].str.strip() != ''].copy()


# ==============================================================================
# 8. MENYIMPAN DATASET BERSIH (SIAP UNTUK PROSES TF-IDF)
# ==============================================================================

df_clean_excel = df_processed.copy()
df_clean_excel['tanggal'] = df_clean_excel['tanggal'].dt.tz_localize(None)

# Simpan CSV dan Excel Bersih
df_clean_excel.to_csv('../dataset/ulasan_shopee_oktober_2026_clean.csv', index=False, encoding='utf-8-sig')
df_clean_excel.to_excel('../dataset/ulasan_shopee_oktober_2026_clean.xlsx', index=False)

print("\n==================================================")
print("       PROSES PRAPEMROSESAN TEKS SELESAI          ")
print("==================================================")
print("File dataset bersih berhasil dibuat!")
print("Lokasi simpan  : ../dataset/")
print("File CSV Bersih: ulasan_shopee_oktober_2026_clean.csv")
print("File Excel Bersih: ulasan_shopee_oktober_2026_clean.xlsx")
print("==================================================")