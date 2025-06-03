from flask import Flask, request, jsonify
import pandas as pd
from keras.models import load_model
from joblib import load

app = Flask(__name__)

@app.route('/')
def home():
    return 'API is working!'

model = load_model('heart_disease.keras')
labelencoder_gender = load('heart_disease_label.pkl')
column_transformer = load('heart_disease_column_transformers.pkl')
scaler = load('heart_disease_standard_scaler.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    expected_keys = ['age', 'sex', 'dataset', 'cp', 'trestbps', 'chol', 'fbs',
                     'restecg', 'thalch', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    if not all(key in data for key in expected_keys):
        return jsonify({'error': 'Missing data'}), 400

    # تحويل البيانات إلى DataFrame
    df = pd.DataFrame([data])

    # معالجة البيانات بنفس الطريقة
    df['sex'] = labelencoder_gender.transform(df['sex'])
    df['fbs'] = df['fbs'].astype(int)
    df['exang'] = df['exang'].astype(int)

    df = column_transformer.transform(df)
    df = scaler.transform(df)

    # التنبؤ
    proba = model.predict(df)[0][0]
    prediction = int(proba > 0.5)

    return jsonify({'prediction': prediction, 'probability': float(proba)})


if __name__ == '__main__':
    app.run(debug=True)
