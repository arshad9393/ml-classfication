from flask import Flask, render_template, request
import pickle
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load trained model and utilities
with open('wine_type_logistic_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

with open('label_encoder.pkl', 'rb') as file:
    label_encoder = pickle.load(file)


# ---------------- ROUTES ---------------- #
@app.route('/')
def home():
    """Render the input form page."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handle form submission and predict wine type."""
    try:
        # Collect input values from form
        features = [
            float(request.form['fixed_acidity']),
            float(request.form['volatile_acidity']),
            float(request.form['citric_acid']),
            float(request.form['residual_sugar']),
            float(request.form['chlorides']),
            float(request.form['free_sulfur_dioxide']),
            float(request.form['total_sulfur_dioxide']),
            float(request.form['density']),
            float(request.form['pH']),
            float(request.form['sulphates']),
            float(request.form['alcohol']),
            float(request.form['quality'])
        ]

        # Prepare for prediction
        features = np.array(features).reshape(1, -1)
        scaled = scaler.transform(features)
        prediction = model.predict(scaled)
        wine_type = label_encoder.inverse_transform(prediction)[0]

        return render_template('result.html',
                               prediction_text=f"🍷 The predicted wine type is: {wine_type}")

    except Exception as e:
        return render_template('result.html',
                               prediction_text=f"⚠️ Error: {str(e)}")


if __name__ == "__main__":
    app.run(debug=True)
