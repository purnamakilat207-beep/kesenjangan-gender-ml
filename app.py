from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Baca data dan model
data = pd.read_csv("data/data_gender_olahan.csv")
paket = joblib.load("model/model_gender.pkl")

model = paket["model"]
fitur = paket["fitur"]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/data")
def data_api():
    kolom = [
        "provinsi",
        "gap_pendidikan",
        "gap_tpak",
        "gap_upah_persen",
        "gap_pekerjaan",
        "kategori_gap"
    ]
    return jsonify(data[kolom].round(2).to_dict(orient="records"))

@app.route("/prediksi", methods=["POST"])
def prediksi():
    isi = request.get_json()

    input_data = pd.DataFrame([{
        "gap_pendidikan": float(isi["gap_pendidikan"]),
        "gap_tpak": float(isi["gap_tpak"]),
        "gap_upah_persen": float(isi["gap_upah_persen"]),
        "rls_perempuan": float(isi["rls_perempuan"]),
        "tpak_perempuan": float(isi["tpak_perempuan"])
    }])

    hasil = model.predict(input_data[fitur])[0]

    return jsonify({"hasil": hasil})

if __name__ == "__main__":
    app.run(debug=True)
