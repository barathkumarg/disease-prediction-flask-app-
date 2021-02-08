import pickle
import csv
import pandas as pd
import collections

model_heart = pickle.load(open('model_heart_disease.pkl','rb'))

#model_diabetics=pickle.load(open('model_diabetics.pkl','rb'))

#output1=(model_heart.predict([[63,1,3,145,233,1,0,150,0,2.3,0,0,1]]))

#output=(model_diabetics.predict([[23,143,12,12,12,12,22]]))

#print(output[0])

#print(output1[0])


df=pd.read_csv('Test Heart_disease dataset.csv')
cols=['name','patient_id','age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']
result_cols=[]
for col in df.columns:
    print(col)
    result_cols.append(col)

if collections.Counter(cols)==collections.Counter(result_cols):
    print("true")


name=df['name'].to_list()
patient_id=df['patient_id'].to_list()
age=df['age'].to_list()
sex=df['sex'].to_list()
cp=df['cp'].to_list()
trestbps=df['trestbps'].to_list()
chol=df['chol'].to_list()
fbs=df['fbs'].to_list()
restecg=df['restecg'].to_list()
thalach=df['thalach'].to_list()
exang=df['exang'].to_list()
oldpeak=df['oldpeak'].to_list()
slope=df['slope'].to_list()
ca=df['ca'].to_list()
thal=df['thal'].to_list()

positive_case=[]
negative_case=[]

for i in range(0,len(df.columns)):
    output1 = (model_heart.predict([[age[i], sex[i], cp[i], trestbps[i], chol[i], fbs[i], restecg[i], thalach[i], exang[i], oldpeak[i], slope[i], ca[i], thal[i]]]))
    output1=output1[0]

    if output1==1:
        positive_case.append(name[i])
        positive_case.append(patient_id[i])
    else:
        negative_case.append(name[i])
        negative_case.append(patient_id[i])
print(positive_case)
#print(negative_case)
for i in range(0,len(positive_case),2):
    print(positive_case[i])
    print(positive_case[i+1])


