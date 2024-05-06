import numpy as np
from flask import Flask, request, render_template
import pickle

# Load the trained classifier
with open('new_gestational.pkl', 'rb') as f:
    model = pickle.load(f)

# Create Flask app
app = Flask(__name__)

# Route to render the HTML form
@app.route("/")
def home():
    return render_template("prediction_form.html")

# Route to handle prediction
@app.route("/predict", methods=["POST"])
def predict():
    # Extract the input values from the form
    features = [float(request.form[field]) for field in ["Age", "NoofPregnancy", "GestationinpreviousPregnancy", "BMI", "HDL", "FamilyHistory", "LargeChildorBirthDefault", "PCOS", "SysBP", "DiaBP", "OGTT", "Hemoglobin", "SedentaryLifestyle", "Prediabetes"]]

    try:
        # Make prediction
        prediction = model.predict([features])[0]

        # Convert prediction to human-readable format
        prediction_text = "Positive" if prediction == 1 else "Negative"
    
        # Return the prediction as a response
        return render_template("prediction_result.html", prediction=prediction_text)

    except Exception as e:
        return f"An error occurred: {e}"

if __name__ == "__main__":
    app.run(debug=True)
