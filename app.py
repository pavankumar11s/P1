import numpy as np
import pickle
from flask import Flask, request, render_template
import csv

# Load ML model
model = pickle.load(open('model.pkl', 'rb'))
model2 = pickle.load(open('model2.pkl', 'rb'))
# Create application
app = Flask(__name__)


# Bind home function to URL
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/h')
def h():
    return render_template('home.html')

@app.route('/simple')
def simple():
    return render_template('simple_predictor.html')

@app.route('/advanced')
def advanced():
    return render_template('advanced_predictor.html')

@app.route('/p_det')
def details():
    results = []
    with open('Patient Details.csv') as csvfile:
        reader = list(csv.DictReader(csvfile))

    for row in reader:
        results.append(dict(row))

    fieldnames = [key for key in results[0].keys()]

    return render_template('details.html', results=results, fieldnames=fieldnames, len=len)


@app.route('/simple', methods=['POST'])
def detect():
    features = [i for i in request.form.values()]
    f2 = [int(i) for i in features[1:]]
    f2[0]*=365
    array_features = np.asarray(f2)
    prediction = model2.predict(array_features.reshape(1, -1))
    output = prediction[0]
    if output == 1:
        return render_template('positive.html',person=features[0])
    else:
        return render_template('negative.html',person=features[0])

# Bind predict function to URL
@app.route('/advanced', methods=['POST'])
def predict():
    # Put all form entries values in a list
    features = [i for i in request.form.values()]
    # Convert features to array
    f2=[float(i) for i in features[1:]]
    array_features = np.asarray(f2)
    # Predict features
    prediction = model.predict(array_features.reshape(1, -1))
    output = prediction[0]
    f3=[]
    # Convert to user format to store in database
    # get date
    from datetime import datetime
    now = datetime.now()
    dt = now.strftime("%d/%m/%Y %H:%M:%S").split()
    f3.append(dt[0])
    f3.append(dt[1])
    f3.append(features[0]) # get name
    f3.append(features[1]) # get age
    if features[2]=='1':
        f3.append('Male')
    else:
        f3.append('Female')     # get gender
    if features[3]=='0':
        f3.append('Typical Angina')
    elif features[3]=='1':
        f3.append('Atypical Angina')
    elif features[3]=='2':
        f3.append('Non-Anginal Pain')
    else:
        f3.append('Asymptomatic')   # get cp type
    f3.append(features[4]) # get RestBP
    f3.append(features[5]) # get Chol
    if features[6]=='1':
        f3.append('Yes')
    else:
        f3.append('No')
    if features[7]=='0':
        f3.append('Normal')
    elif features[7]=='1':
        f3.append('Having ST-T wave abnormality')
    else:
        f3.append('Probable or definite left ventricular hypertrophy') # get RestECG
    f3.append(features[8]) # get Thalach
    if features[9]=='1':
        f3.append('Yes')
    else:
        f3.append('No') # get exang
    if output == 1:
        f3.append('Positive')
    else:
        f3.append('Negative') # get result
    f3.append(features[10]) # get oldpeak
    if features[11]=='0':
        f3.append('Up')
    elif features[11]=='1':
        f3.append('Flat')
    else:
        f3.append('Down') # get slope
    f3.append(features[12]) # get ca
    if features[13]=='0':
        f3.append('Normal')
    elif features[13]=='1':
        f3.append('Fixed Defect')
    else:
        f3.append('Reversable Defect') #get thal
    from csv import writer
    # Save Patient Data
    with open("Patient Details.csv", 'a', newline='') as f:
        writer_object = writer(f)
        writer_object.writerow(f3)
    f.close()
    # Check the output values and retrive the result with html tag based on the value
    if output == 1:
        return render_template('positive.html',person=features[0])
    else:
        return render_template('negative.html',person=features[0])



if __name__ == '__main__':
    # Run the application
    app.run()