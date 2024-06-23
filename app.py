from flask import Flask, request, jsonify
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from dotenv import load_dotenv
from pyngrok import ngrok
import os

load_dotenv()

app = Flask(__name__)

# Load the trained machine learning model
with open('models/RandomForest.pkl', 'rb') as file:
    model = pickle.load(file)

# Mapping dictionaries
age_mapping = {'25-30': 0, '30-35': 0.25, '35-40': 0.5, '40-45': 0.75, '45-50': 1}
binary_mapping = {'Yes': 1, 'No': 0}
frequency_mapping = {'Not at all': 0, 'Sometimes': 0.5, 'Often': 1, 'Two or more days a week': 0.5, 'Maybe': 0.5, 'Not interested to say': 0.5}

@app.route('/predictDepression', methods=['POST'])
def predict():
    if request.method == 'POST':
        data = request.get_json()
        processed_data = preprocess_data(pd.DataFrame(data, index=[0]))

        input_data = processed_data[['Age', 'Irritable towards baby & partner', 'Trouble sleeping at night',
                                     'Problems concentrating or making decision', 'Overeating or loss of appetite',
                                     'Feeling anxious', 'Feeling of guilt', 'Problems of bonding with baby',
                                     'Suicide attempt']]

        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)[:, 1]

        return jsonify({'prediction': int(prediction[0]), 'probability_percentage': round(probability[0] * 100, 2)})
    else:
        return jsonify({'error': 'Method not allowed'})

def preprocess_data(data):
    data['Age'] = data['Age'].map(age_mapping)
    data['Feeling anxious'] = data['Feeling anxious'].map(binary_mapping)
    data['Trouble sleeping at night'] = data['Trouble sleeping at night'].map(frequency_mapping)
    data['Problems concentrating or making decision'] = data['Problems concentrating or making decision'].map(frequency_mapping)
    data['Irritable towards baby & partner'] = data['Irritable towards baby & partner'].map(frequency_mapping)
    data['Overeating or loss of appetite'] = data['Overeating or loss of appetite'].map(frequency_mapping)
    data['Feeling of guilt'] = data['Feeling of guilt'].map(frequency_mapping)
    data['Problems of bonding with baby'] = data['Problems of bonding with baby'].map(frequency_mapping)
    data['Suicide attempt'] = data['Suicide attempt'].map(frequency_mapping)
    return data

if __name__ == "__main__":
    NGROK_AUTH = os.getenv("NGROK_AUTH")
    if not NGROK_AUTH:
        raise ValueError("NGROK_AUTH environment variable is not set.")

    port = 5000
    ngrok.set_auth_token(NGROK_AUTH)
    ngrok_tunnel = ngrok.connect(port, domain="owl-infinite-usefully.ngrok-free.app")
    print("Public URL:", ngrok_tunnel.public_url)
    app.run(host="0.0.0.0", port=port, threaded=True)
