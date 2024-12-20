import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    #Extraction des entrées utilisateur, conversion des valeurs en float
    float_features = [float(x) for x in request.form.values()]
    final_features = [np.array(float_features)]
    
    #Calcul des probabilités
    probabilities = model.predict_proba(final_features)
    probability_of_default = round(probabilities[0][1] * 100, 2)  
    probability_of_no_default = round(probabilities[0][0] * 100, 2)  
    
    #Prédiction binaire (0 ou 1)
    prediction = model.predict(final_features)[0]
    result = "fera défaut" if prediction == 1 else "ne fera pas défaut"
    
    #Message de sortie
    output_message = f"Le client {result} avec une probabilité de {probability_of_default}%."
    
    return render_template('index.html', prediction_text=output_message)

if __name__ == "__main__":
    app.run(debug=True)
