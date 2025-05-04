from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load models once when the server starts
best_model = joblib.load("best_anxiety_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")
top_five_questions = joblib.load("top_five_questions.pkl")

# Map string answers to numeric scale
answer_map = {
    "Never": 0,
    "Almost Never": 1,
    "Sometimes": 2,
    "Fairly Often": 3,
    "Very Often": 4
}

# Recommendation text
recommendations = {
    "Minimal": "You're doing well. Maintain a healthy balance and keep practicing self-care.",
    "Mild": "Try taking breaks and practicing light mindfulness exercises.",
    "Moderate": "Consider talking to a counselor or using mindfulness apps.",
    "Severe": "You should consult a mental health professional soon."
}

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        answers = data.get("answers", {})  # Expecting a map of q1 to q5 like Firebase

        if len(answers) != 5:
            return jsonify({"error": "Expected 5 answers"}), 400

        # Map q1, q2,... to actual questions in the same order
        mapped_answers = {}
        for idx, q in enumerate(top_five_questions):
            key = f"q{idx+1}"
            val = answers.get(key)
            if val not in answer_map:
                return jsonify({"error": f"Invalid value '{val}' for {key}"}), 400
            mapped_answers[q] = answer_map[val]

        df = pd.DataFrame([mapped_answers])
        prediction = best_model.predict(df)
        label = label_encoder.inverse_transform(prediction)[0]
        recommendation = recommendations.get(label, "Take care of your mental well-being.")

        return jsonify({
            "prediction": label,
            "recommendation": recommendation
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/", methods=["GET"])
def home():
    return "Mental Wellness API is running"
