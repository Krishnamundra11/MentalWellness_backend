from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load model and support files once before first request
@app.before_first_request
def load_model():
    global model, le, top_five_questions
    model = joblib.load('best_anxiety_model.pkl')
    le = joblib.load('label_encoder.pkl')
    top_five_questions = joblib.load('top_five_questions.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    # Extract and prepare user responses
    user_responses = {question: int(data.get(question, 0)) for question in top_five_questions}
    user_data = pd.DataFrame([user_responses])
    
    # Make prediction
    prediction = model.predict(user_data)
    prediction_label = le.inverse_transform(prediction)[0]
    
    # Predict probabilities (if supported)
    probabilities = {}
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(user_data)[0]
        class_labels = le.classes_
        probabilities = {str(class_labels[i]): float(prob) for i, prob in enumerate(probs)}
    
    # Basic guidance message
    if prediction_label == "Minimal":
        guidance = "Minimal anxiety."
    elif prediction_label == "Mild":
        guidance = "Mild anxiety."
    elif prediction_label == "Moderate":
        guidance = "Moderate anxiety."
    else:  # Severe
        guidance = "Severe anxiety. Consider professional support."

    return jsonify({
        "prediction": prediction_label,
        "probabilities": probabilities,
        "guidance": guidance
    })

if __name__ == '__main__':
    app.run(debug=True)
