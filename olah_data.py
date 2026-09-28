import pandas as pd

# 1. Baca data mentah
data = pd.read_csv("data/data_gender_2024.csv")

# 2. Cek data kosong
print("Data kosong:")
print(data.isnull().sum())

# 3. Buat variabel gap
data["gap_pendidikan"] = (
    data["rls_laki_laki"] - data["rls_perempuan"]
)

data["gap_tpak"] = (
    data["tpak_laki_laki"] - data["tpak_perempuan"]
)

data["gap_pekerjaan"] = (
    data["epr_laki_laki"] - data["epr_perempuan"]
)

# 4. Hitung persentase gap upah
data["gap_upah_persen"] = (
    (data["upah_laki_laki"] - data["upah_perempuan"])
    / data["upah_laki_laki"] * 100
)

# 5. Buat label yang akan dipelajari Machine Learning
#    <=20  = Rendah
#    20-30 = Sedang
#    >30   = Tinggi
def kategori(gap):
    if gap <= 20:
        return "Rendah"
    elif gap <= 30:
        return "Sedang"
    else:
        return "Tinggi"

data["kategori_gap"] = data["gap_pekerjaan"].apply(kategori)

# 6. Bulatkan angka agar rapi
angka = data.select_dtypes(include="number").columns
data[angka] = data[angka].round(2)

# 7. Simpan hasil olahan
data.to_csv("data/data_gender_olahan.csv", index=False)

print("\nData sudah diolah.")
print(data.head())
print("\nJumlah tiap kategori:")
print(data["kategori_gap"].value_counts())
