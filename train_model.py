import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Baca data yang sudah diolah
data = pd.read_csv("data/data_gender_olahan.csv")

# Fitur yang dipakai AI
fitur = [
    "gap_pendidikan",
    "gap_tpak",
    "gap_upah_persen",
    "rls_perempuan",
    "tpak_perempuan"
]

# Target yang ingin diprediksi
target = "kategori_gap"

X = data[fitur]
y = data[target]

# Bagi data: 80% latihan, 20% pengujian
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Buat model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

# Training
model.fit(X_train, y_train)

# Pengujian
prediksi = model.predict(X_test)

print("=== HASIL MACHINE LEARNING ===")
print("Akurasi:", round(accuracy_score(y_test, prediksi) * 100, 2), "%")
print("\nLaporan klasifikasi:")
print(classification_report(y_test, prediksi, zero_division=0))

# Simpan model
joblib.dump(
    {"model": model, "fitur": fitur},
    "model/model_gender.pkl"
)

print("\nModel berhasil disimpan di model/model_gender.pkl")
