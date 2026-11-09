import pandas as pd
import time
import os
from google_play_scraper import reviews, Sort

# =====================================
# 1. KONFIGURASI
# =====================================

APP_ID = 'com.shopee.id'

START_DATE = pd.Timestamp(
    '2026-10-01',
    tz='Asia/Jakarta'
)

END_DATE = pd.Timestamp(
    '2027-01-01',
    tz='Asia/Jakarta'
)

all_reviews = []
continuation_token = None

print("=====================================")
print("       SCRAPING ULASAN SHOPEE")
print("=====================================")
print("Tanggal mulai :", START_DATE)
print("Tanggal akhir : 31 Desember 2026")
print("Memulai proses scraping...\n")


# =====================================
# 2. FUNGSI KONVERSI TANGGAL WIB
# =====================================

def convert_to_wib(date):

    date = pd.Timestamp(date)

    if date.tzinfo is None:
        return date.tz_localize('Asia/Jakarta')
    else:
        return date.tz_convert('Asia/Jakarta')


# =====================================
# 3. WEB SCRAPING
# =====================================

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
    batch_dates = [
        convert_to_wib(r['at'])
        for r in result
    ]

    # Mencari tanggal ulasan tertua
    oldest_date = min(batch_dates)

    # Mengambil ulasan sesuai periode penelitian
    for r, review_date in zip(result, batch_dates):

        if START_DATE <= review_date < END_DATE:
            all_reviews.append(r)

    print(
        f"Data tahun 2026: {len(all_reviews)} | "
        f"Ulasan tertua: {oldest_date}"
    )

    # Berhenti jika sudah melewati tanggal awal
    if oldest_date < START_DATE:
        print("\nBatas tanggal awal tercapai.")
        break

    # Berhenti jika tidak ada halaman berikutnya
    if continuation_token is None:
        print("\nSeluruh data yang tersedia telah diperiksa.")
        break

    time.sleep(1.5)


# =====================================
# 4. MEMBENTUK DATASET
# =====================================

df = pd.DataFrame(all_reviews)

if df.empty:
    raise ValueError(
        "Tidak ditemukan ulasan pada tahun 2026. "
        "Periksa ketersediaan data historis."
    )

# Menghapus data duplikat
df = df.drop_duplicates(subset='reviewId')

# Memilih kolom yang diperlukan
df = df[[
    'reviewId',
    'userName',
    'content',
    'score',
    'at',
    'thumbsUpCount'
]]

# Mengganti nama kolom
df.columns = [
    'id_ulasan',
    'nama_pengguna',
    'ulasan',
    'rating',
    'tanggal',
    'jumlah_like'
]

# Mengubah tanggal ke WIB
df['tanggal'] = df['tanggal'].apply(convert_to_wib)


# =====================================
# 5. FILTER DATA TAHUN 2026
# =====================================

df_filtered = df[
    (df['tanggal'] >= START_DATE) &
    (df['tanggal'] < END_DATE)
].copy()

# Menghapus ulasan kosong
df_filtered = df_filtered.dropna(
    subset=['ulasan']
)

df_filtered = df_filtered[
    df_filtered['ulasan'].str.strip() != ''
].copy()

if df_filtered.empty:
    raise ValueError(
        "Tidak ada ulasan valid setelah seleksi data."
    )


# =====================================
# 6. PELABELAN SENTIMEN OTOMATIS
# =====================================

def labeling(rating):

    if rating <= 2:
        return 'Negatif'

    elif rating == 3:
        return 'Netral'

    else:
        return 'Positif'


df_filtered['sentimen'] = (
    df_filtered['rating'].apply(labeling)
)


# =====================================
# 7. MENYIMPAN DATASET
# =====================================

os.makedirs('dataset', exist_ok=True)

# Mengurutkan data berdasarkan tanggal
df_filtered = df_filtered.sort_values(
    by='tanggal'
)

# Menghapus informasi timezone saat menyimpan Excel
df_excel = df_filtered.copy()

df_excel['tanggal'] = (
    df_excel['tanggal'].dt.tz_localize(None)
)

# Menyimpan CSV
df_excel.to_csv(
    'dataset/ulasan_shopee_2026.csv',
    index=False,
    encoding='utf-8-sig'
)

# Menyimpan Excel
df_excel.to_excel(
    'dataset/ulasan_shopee_2026.xlsx',
    index=False
)


# =====================================
# 8. INFORMASI HASIL SCRAPING
# =====================================

print("\n=====================================")
print("           HASIL SCRAPING")
print("=====================================")

print(
    "Total ulasan tahun 2026:",
    len(df_filtered)
)

print("\nRentang tanggal dataset:")

print(
    "Tanggal paling awal:",
    df_filtered['tanggal'].min()
)

print(
    "Tanggal paling akhir:",
    df_filtered['tanggal'].max()
)

print("\nDistribusi Sentimen:")

print(
    df_filtered['sentimen'].value_counts()
)

print("\nDistribusi Rating:")

print(
    df_filtered['rating'].value_counts().sort_index()
)

print("\nDataset berhasil disimpan!")
print("Lokasi: folder dataset/")

print("File CSV   : ulasan_shopee_2026.csv")
print("File Excel : ulasan_shopee_2026.xlsx")

print("=====================================")