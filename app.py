from flask import Flask, request, jsonify, render_template
import os
import sys
import numpy as np
import pickle
from werkzeug.utils import secure_filename
import pandas as pd
import collections

model_diabetics = pickle.load(open('model_diabetics.pkl', 'rb'))

model_heart=pickle.load(open('model_heart_disease.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/heart_disease')
def heart_disease():
    return render_template('heart_test_for_alone.html')

@app.route('/diabetics')
def diabetics():
    return render_template('diabetics_test_for_alone.html')

@app.route('/heart_disease_many')
def heart_disease_many():
    return render_template('heart_test_for_many.html')
@app.route('/diabetics_disease_many')
def diabetics_disease_many():
    return render_template('diabetics_test_for_many.html')

@app.route('/contact')
def contact():
    return render_template('information.html')

@app.route('/predict_diabetics_alone',methods=['GET', 'POST'])
def predict_diabetics_alone():
    if request.method == 'POST':
        age = int(request.form['age'])
        glucose=float(request.form['glucose'])
        pressure = float(request.form['pressure'])
        skin = float(request.form['skin'])
        bmi=float(request.form['bmi'])
        pedi=float(request.form['pedi'])

        int_features = []

        int_features.append(glucose)
        int_features.append(pressure)
        int_features.append(skin)
        int_features.append(bmi)
        int_features.append(pedi)
        int_features.append(age)
        final_features = [np.array(int_features)]
        prediction = model_diabetics.predict(final_features)

        prediction=prediction[0]
        if (prediction==0):
            res="Test result is negative, Patient have no diabetics"
            clr="green"
        else:
            res="Test result is positive, Patient have diabetics"
            clr="red"


    return render_template('diabetics_test_for_alone.html',res=res,clr=clr)

@app.route('/predict_diabetics_many',methods=['GET', 'POST'])
def predict_diabetics_many():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, secure_filename(f.filename))

        f.save(file_path)
        df = pd.read_csv(file_path)

        cols = ['name', 'patient_id', 'age', 'Plas', 'Pres', 'skin', 'mass', 'pedi']
        result_cols = []
        for col in df.columns:

            result_cols.append(col)

        if collections.Counter(cols) == collections.Counter(result_cols):

            name = df['name'].to_list()
            patient_id = df['patient_id'].to_list()
            age = df['age'].to_list()
            Plas=df['Plas'].to_list()
            Pres=df['Pres'].to_list()
            skin=df['skin'].to_list()
            mass=df['mass'].to_list()
            pedi=df['pedi'].to_list()
            positive_case = []
            negative_case = []


            for i in range(0, len(df.index)):

                output1 = (model_diabetics.predict([[Plas[i], Pres[i], skin[i], mass[i], pedi[i],age[i]]]))
                output1 = output1[0]

                if output1 == 1:
                    positive_case.append(name[i] +'  -'+ patient_id[i])

                else:
                    negative_case.append(name[i]+'  -'+ patient_id[i])


    return render_template('diabetics_test_for_many.html',negative=negative_case, positive=positive_case,a="Positive Cases [Affected]",b="Negative Cases [Unaffected]")

@app.route('/predict_heart_alone',methods=['GET', 'POST'])
def predict_heart_alone():
    if request.method == 'POST':
        age = int(request.form['age'])
        gender = int(request.form['gender'])
        cp = int(request.form['cp'])
        rbp= int(request.form['rbp'])
        chol = int(request.form['chol'])
        fpr = int(request.form['fpr'])
        ecr = int(request.form['ecr'])
        mx_heart = int(request.form['mx_heart'])
        eia = int(request.form['eia'])
        op = int(request.form['op'])
        st= int(request.form ['st'])
        vf = int(request.form['vf'])
        thal = int(request.form['thal'])

        int_features = []
        int_features.append(age)
        int_features.append(gender)
        int_features.append(cp)
        int_features.append(rbp)
        int_features.append(chol)
        int_features.append(fpr)
        int_features.append(ecr)
        int_features.append(mx_heart)
        int_features.append(eia)
        int_features.append(op)
        int_features.append(st)
        int_features.append(vf)
        int_features.append(thal)

        final_features = [np.array(int_features)]
        prediction = model_heart.predict(final_features)

        prediction = prediction[0]
        if (prediction == 0):
            res = "Test result is negative, Patient have no Heart Disease"
            clr = "green"
        else:
            res = "Test result is positive, Patient have Heart Disease"
            clr = "red"
    return render_template('heart_test_for_alone.html', res=res, clr=clr)

@app.route('/predict_heart_many',methods=['GET', 'POST'])
def predict_heart_many():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, secure_filename(f.filename))

        f.save(file_path)
        df = pd.read_csv(file_path)
        cols = ['name', 'patient_id', 'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang',
                'oldpeak', 'slope', 'ca', 'thal']
        result_cols = []
        for col in df.columns:
            print(col)
            result_cols.append(col)

        if collections.Counter(cols) == collections.Counter(result_cols):
            print('True')


            name = df['name'].to_list()
            patient_id = df['patient_id'].to_list()
            age = df['age'].to_list()
            sex = df['sex'].to_list()
            cp = df['cp'].to_list()
            trestbps = df['trestbps'].to_list()
            chol = df['chol'].to_list()
            fbs = df['fbs'].to_list()
            restecg = df['restecg'].to_list()
            thalach = df['thalach'].to_list()
            exang = df['exang'].to_list()
            oldpeak = df['oldpeak'].to_list()
            slope = df['slope'].to_list()
            ca = df['ca'].to_list()
            thal = df['thal'].to_list()

            positive_case = []
            negative_case = []

            for i in range(0, len(df.index)):
                output1 = (model_heart.predict([[age[i], sex[i], cp[i], trestbps[i], chol[i], fbs[i], restecg[i],
                                             thalach[i], exang[i], oldpeak[i], slope[i], ca[i], thal[i]]]))
                output1 = output1[0]

                if output1 == 1:
                    positive_case.append(name[i] + '  -' + patient_id[i])

                else:
                    negative_case.append(name[i] + '  -' + patient_id[i])



    return render_template('heart_test_for_many.html', negative=negative_case, positive=positive_case,
                           a="Positive Cases [Affected]", b="Negative Cases [Unaffected]")


if __name__=='__main__':
    app.run(debug=True)