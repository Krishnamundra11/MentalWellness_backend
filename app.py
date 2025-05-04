import joblib
import pandas as pd

print("\nModel, label encoder, and important questions saved for future predictions.")

# Function to get user input and make predictions
def predict_anxiety_level():
    print("\n" + "=" * 50)
    print("ANXIETY PREDICTION SYSTEM")
    print("=" * 50)
    print("\nPlease answer the following questions on a scale of 0-4:")
    print("0: Not at all")
    print("1: Several days")
    print("2: More than half the days")
    print("3: Nearly every day")
    print("4: Every day")

    user_responses = {}
    for i, question in enumerate(top_five_questions, 1):
        short_q = question.split('?')[0].strip() + "?"
        while True:
            try:
                response = int(input(f"\nQ{i}: {short_q}\nYour answer (0-4): "))
                if 0 <= response <= 4:
                    user_responses[question] = response
                    break
                else:
                    print("Please enter a number between 0 and 4.")
            except ValueError:
                print("Please enter a valid number.")

    # Create a DataFrame from user input
    user_data = pd.DataFrame([user_responses])

    # Make prediction
    prediction = best_model.predict(user_data)
    prediction_label = label_encoder.inverse_transform(prediction)[0]

    # Calculate total score
    total_score = sum(user_responses.values())
    max_possible = len(top_five_questions) * 4

    # Dictionary of recommendations based on anxiety levels
    recommendations = {
        "Minimal": "You're doing well. Maintain a healthy study-life balance and continue self-care practices.",
        "Mild": "Try incorporating short breaks, light exercise, or breathing exercises into your routine.",
        "Moderate": "Consider talking to a counselor or using mindfulness apps to manage anxiety.",
        "Severe": "It is advisable to consult a mental health professional. Prioritize your well-being over academic pressure."
    }

    # Get recommendation
    recommendation = recommendations.get(prediction_label, "Stay mindful of your mental health. Seek support if needed.")

    print("\n" + "=" * 50)
    print(f"Total Score: {total_score}/{max_possible}")
    print(f"Predicted Anxiety Level: {prediction_label}")
    print(f"Recommendation: {recommendation}")
    print("=" * 50)

    print("\nNote: This is not a clinical diagnosis. If you're concerned about your mental health,")
    print("please consult with a qualified healthcare professional.")

    # Optional: return values if integrating with an API
    return {
        "prediction": prediction_label,
        "recommendation": recommendation,
        
        "answers": user_responses
    }

# Main runner function
def run_anxiety_prediction_system():
    try:
        global best_model, label_encoder, top_five_questions
        best_model = joblib.load('best_anxiety_model.pkl')
        label_encoder = joblib.load('label_encoder.pkl')
        top_five_questions = joblib.load('top_five_questions.pkl')

        # Call the prediction function
        result = predict_anxiety_level()
        return result  # for API use if needed

    except FileNotFoundError:
        print("Error: Model files not found. Please run the training script first.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    run_anxiety_prediction_system()
