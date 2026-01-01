from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("candida_auris_dynamics_model.pkl")

PU = 200
PC = 30
PI = 10
HC = 15
HU = 85

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        form = request.form

        Lambda = float(form.get("Lambda", 0))
        beta1 = float(form.get("beta1", 0))
        beta2 = float(form.get("beta2", 0))
        d1 = float(form.get("d1", 0))
        d2 = float(form.get("d2", 0))
        lam = float(form.get("lambda", 0))
        inv_sigma = float(form.get("1_over_sigma", 1))
        phi = float(form.get("phi", 0))
        mu = float(form.get("mu", 0))

        icu_days = int(form.get("icu_days", 0))
        hcw_contact = form.get("hcw_contact", "low")
        immune_status = form.get("immune_status", "normal")

        sigma = 1 / inv_sigma

        dPU_dt = Lambda - beta2 * HC * PU - d1 * PU + phi * (PC + PI)
        dPC_dt = beta2 * HC * PU - (d2 + phi + sigma) * PC
        dHC_dt = beta1 * PC * HU - lam * HC

        features = np.array([[ 
            Lambda, beta1, beta2,
            d1, d2, lam,
            inv_sigma, phi, mu,
            dPU_dt, dPC_dt, dHC_dt
        ]])

        dPI_dt = float(model.predict(features)[0])

        if dPI_dt < 40:
            icu_risk = "LOW"
            base_score = 1
        elif dPI_dt < 80:
            icu_risk = "MODERATE"
            base_score = 2
        else:
            icu_risk = "HIGH"
            base_score = 3

        score = base_score

        if icu_days > 7:
            score += 1
        if hcw_contact == "medium":
            score += 0.5
        if hcw_contact == "high":
            score += 1
        if immune_status == "immunocompromised":
            score += 1

        if score <= 2:
            individual_risk = "LOW"
        elif score <= 4:
            individual_risk = "MODERATE"
        else:
            individual_risk = "HIGH"

        return jsonify({
            "dPI_dt": round(dPI_dt, 4),
            "icu_risk": icu_risk,
            "individual_risk": individual_risk
        })

    except Exception as e:
        return jsonify({
            "dPI_dt": None,
            "icu_risk": "LOW",
            "individual_risk": "LOW",
            "error": str(e)
        }), 200

if __name__ == "__main__":
    app.run(debug=True)
