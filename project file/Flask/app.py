from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load model and scaler
model = pickle.load(open("rf_stage.pkl", "rb"))
scaler = pickle.load(open("normalizer.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def predict():
    prediction_text = None
    description_text = None

    # Default values (these are just examples; replace with real means/medians from your dataset)
    default_values = [
        0,    # N_Days
        50,   # Age
        0,    # Sex
        0,    # Ascites
        0,    # Hepatomegaly
        0,    # Spiders
        0,    # Edema
        1.2,  # Bilirubin
        200,  # Cholesterol
        3.5,  # Albumin
        100,  # Copper
        120,  # Alk_Phos
        60,   # SGOT
        150,  # Triglycerides
        250,  # Platelets
        10,   # Prothrombin
        1     # Stage (optional, but model expects 17 values)
    ]

    if request.method == "POST":
        try:
            features = []
            for i in range(17):
                raw_val = request.form.get(f"f{i+1}", "").strip()
                if raw_val == "":
                    features.append(default_values[i])
                else:
                    features.append(float(raw_val))

            features = np.array(features).reshape(1, -1)
            features_scaled = scaler.transform(features)
            prediction = model.predict(features_scaled)[0]

            stage_map = {
                0: "Stage 1 (Early Stage)",
                1: "Stage 2 (Moderate Stage)",
                2: "Stage 3 (Advanced Cirrhosis)"
            }
            desc_map = {
                0: "Liver is beginning to show signs of cirrhosis. Early diagnosis and treatment can prevent progression.",
                1: "Moderate damage to the liver. Medical intervention is needed to manage symptoms and prevent complications.",
                2: "Severe liver damage detected. Immediate and ongoing medical care is essential."
            }

            prediction_text = f"Predicted Liver Disease Stage: {stage_map[prediction]}"
            description_text = desc_map[prediction]

        except Exception as e:
            prediction_text = f"An error occurred: {str(e)}"
            description_text = ""

    return render_template("index.html", prediction_text=prediction_text, description_text=description_text)

if __name__ == "__main__":
    app.run(debug=True)
