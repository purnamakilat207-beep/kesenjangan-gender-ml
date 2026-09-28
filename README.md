# PROYEK ANALISIS KESENJANGAN GENDER DENGAN MACHINE LEARNING

Judul:
Analisis Kesenjangan Gender dalam Pendidikan dan Pekerjaan Menggunakan Machine Learning

## 1. Urutan pengerjaan

1. data_gender_2024.csv = data awal
2. olah_data.py = menghitung gap dan membuat kategori
3. data_gender_olahan.csv = data siap untuk Machine Learning
4. train_model.py = melatih Random Forest
5. model_gender.pkl = model hasil training
6. app.py = menghubungkan model dengan website
7. templates/index.html = tampilan website

## 2. Install

Buka Terminal VS Code:

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt

## 3. Olah data

python olah_data.py

## 4. Training Machine Learning

python train_model.py

## 5. Jalankan website

python app.py

Buka:
http://127.0.0.1:5000

## 6. Logika sederhana

Gap pendidikan = RLS laki-laki - RLS perempuan

Gap TPAK = TPAK laki-laki - TPAK perempuan

Gap pekerjaan = EPR laki-laki - EPR perempuan

Gap upah (%) =
((upah laki-laki - upah perempuan) / upah laki-laki) x 100

Kategori:
0-20   = Rendah
>20-30 = Sedang
>30    = Tinggi

Machine Learning:
Random Forest Classifier.

Catatan:
Dataset tingkat provinsi hanya memiliki 38 observasi, sehingga proyek ini ditujukan untuk pembelajaran/eksplorasi. Jangan menyimpulkan hubungan sebab-akibat dari model.
