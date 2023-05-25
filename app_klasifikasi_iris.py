from flask import Flask, request, render_template
import pickle
import pandas as pd 
import numpy as np 

app = Flask(__name__)

model_file = open('modeliris.pkl', 'rb')
model = pickle.load(model_file, encoding='bytes')

@app.route('/')
def index():
    return render_template('index.html', hasil=0)

@app.route('/predict', methods=['POST'])
def predict():
    '''
    Predict the insurance cost based on user inputs
    and render the result to the html page
    '''
    sepal_length=float(request.form['sepal_length'])
    
    sepal_width=float(request.form['sepal_width'])

    petal_length=float(request.form['petal_length'])

    petal_width=float(request.form['petal_width'])

    x=np.array([[sepal_length,sepal_width,petal_length,petal_width]])

 
    
    prediction = model.predict(x)
    output = round(prediction[0],0)
    if (output==0):
        kelas="setosa"
    elif (output==1):
        kelas="versicolor"
    else:
        kelas="virginica"


    return render_template('index.html', hasil=kelas, sepal_length=sepal_length, sepal_width=sepal_width, petal_length=petal_length, petal_width=petal_width)


if __name__ == '__main__':
    app.run(debug=True)