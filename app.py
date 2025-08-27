from flask import Flask, render_template, request
import pickle
import pandas as pd

model, scaler = pickle.load(open("pcod_model.pkl", "rb"))

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        age = int(request.form["age"])
        height = float(request.form["height"])
        weight = float(request.form["weight"])
        unusual = request.form["unusual"].lower()
        unusual = 1 if unusual in ["yes", "1", "true"] else 0

        bmi = round(weight / ((height / 100) ** 2), 2)

        X_input = pd.DataFrame([[age, height, weight, unusual]],
                               columns=["Age", "Height", "Weight", "Unusual_Bleeding"])

        X_input_scaled = scaler.transform(X_input)

        prediction = model.predict(X_input_scaled)[0]

        if prediction == 1:
            result = f"High chance of PCOD (BMI: {bmi})"
        else:
            result = f"Low chance of PCOD (BMI: {bmi})"

        return render_template("result.html", result=result, bmi=bmi)

    except Exception as e:
        return render_template("result.html", result=f"Error: {str(e)}", bmi="--")

if __name__ == "__main__":
    app.run(debug=True)
