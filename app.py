
from flask import Flask, request, jsonify
import pickle

# Load model and encoders
with open("model.pkl", "rb") as f:
    model = pickle.load(f)
with open("le_stream.pkl", "rb") as f:
    le_stream = pickle.load(f)
with open("le_subject.pkl", "rb") as f:
    le_subject = pickle.load(f)
with open("le_interest.pkl", "rb") as f:
    le_interest = pickle.load(f)
with open("le_career.pkl", "rb") as f:
    le_career = pickle.load(f)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    try:
        # Get and normalize inputs
        stream = data['stream'].strip().title()
        subject = data['best_subject'].strip().title()
        interest = data['interest'].strip().title()

        # Encode inputs
        stream_enc = le_stream.transform([stream])[0]
        subject_enc = le_subject.transform([subject])[0]
        interest_enc = le_interest.transform([interest])[0]

        # Predict
        prediction = model.predict([[stream_enc, subject_enc, interest_enc]])
        career = le_career.inverse_transform(prediction)[0]

        return jsonify({'career': career})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/', methods=['GET'])
def home():
    return "Career Recommender API is running!"

if __name__ == '__main__':
    app.run(debug=True)
