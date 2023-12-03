import os
from flask import Flask, render_template, request

from mlProject.pipeline.prediction import PredictionPipeline

import numpy as np
import pandas as pd


app = Flask(__name__) # create an app instance

@app.route("/", methods=['GET', 'POST']) # Homepage
def home():
    return render_template("index.html")


@app.route('/train', methods=['GET'])
def train():
    os.system('python main.py')
    return render_template('train.html')


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        try:
            field_names = [
                'allelectrons_Total', 'density_Total', 'allelectrons_Average', 'val_e_Average',
                'atomicweight_Average', 'ionenergy_Average', 'el_neg_chi_Average', 
                'R_vdw_element_Average', 'R_cov_element_Average', 'zaratio_Average', 'density_Average'
            ]
            data = []

            for field_name in field_names:
                value = float(request.form[field_name])
                data.append(value)

            matrix = np.array(data).reshape(1, -1)

            data = pd.DataFrame(matrix, columns=field_names)

            prediction_pipeline = PredictionPipeline()
            prediction = prediction_pipeline.predict(data)
            prediction = round(prediction[0], 2)

            return render_template('result.html', prediction_text='The predicted Mohs Hardness is {}'.format(prediction))

        except Exception as e:
            print(e)
            return render_template('result.html', prediction_text='Error occurred. Please ensure all values are correct otherwise report error.')
    
    else:
        return render_template('index.html')


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080) # run the flask app
