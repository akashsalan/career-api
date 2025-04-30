
from flask import Flask, request, jsonify
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

# Load dataset and train model at startup
df = pd.read_csv("student_career_351_final_fixed.csv")

# Label encode columns
le_stream = LabelEncoder()
le_subject = LabelEncoder()
le_interest = LabelEncoder()
le_career = LabelEncoder()

df['Stream_enc'] = le_stream.fit_transform(df['Stream'])
df['Subject_enc'] = le_subject.fit_transform(df['Best_Subject'])
df['Interest_enc'] = le_interest.fit_transform(df['Interest'])
df['Career_enc'] = le_career.fit_transform(df['Career'])

# Prepare features and target
X = df[['Stream_enc', 'Subject_enc', 'Interest_enc']]
y = df['Career_enc']

# Train model
model = DecisionTreeClassifier()
model.fit(X, y)

# Setup Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return "Career Recommender API is running (model trained live)."

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        stream = data['stream'].strip().title()
        subject = data['best_subject'].strip().title()
        interest = data['interest'].strip().title()

        # Encode inputs
        s = le_stream.transform([stream])[0]
        sub = le_subject.transform([subject])[0]
        intr = le_interest.transform([interest])[0]

        # Predict
        pred = model.predict([[s, sub, intr]])
        career = le_career.inverse_transform(pred)[0]

        return jsonify({'career': career})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # default to 5000 locally
    app.run(host="0.0.0.0", port=port, debug=True)

