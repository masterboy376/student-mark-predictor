# coding: utf-8

import numpy as np
import pandas as pd
from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

model = joblib.load("student_mark_predictor.pkl")

df = pd.DataFrame()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global df

    input_features = [int(x) for x in request.form.values()]
    feature_values = np.array(input_features)

    # validate input hours
    if(input_features[0]<0 or input_features[0]>24):
        return render_template('index.html', prediction_text='please enter a valid input between 0 and 24!')
    
    output = model.predict([feature_values])[0][0].round(2)

    # input and predicted value store in df then save as csv file
    df = pd.concat([df,pd.DataFrame({'Study Hours':input_features, 'Predicted Output': [output] })], ignore_index=True)
    print(df)
    df.to_csv('smp_data_from_app.csv')

    return render_template('index.html', prediction_text=f"Your expected mark is {output} according to your study hours.")

app.run(debug=True)