import pandas as pd
import numpy as np
import pickle
 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
 
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from flask import Flask
from flask import request
from flask import jsonify

import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('bank-full.csv', delimiter=';')
df.y = (df.y == 'yes').astype(int)
y = df.y.values

features = ['job', 'duration', 'poutcome']
dicts = df[features].to_dict(orient='records')

dv = DictVectorizer(sparse=False)
X = dv.fit_transform(dicts)

model = LogisticRegression().fit(X, y)

def load(filename: str):
    with open(filename, 'rb') as f_in:
        return pickle.load(f_in)
    
dv = load('dv.bin')
model1 = load('model2.bin')

app = Flask('get-subscription')

@app.route('/predict', methods=['POST'])
def predict():
    # json = Python dictionary
    client = request.get_json()
 
    X = dv.transform([client])
    model.predict_proba(X)
    y_pred = model.predict_proba(X)[0,1] 
    subscription = y_pred >= 0.5
 
    result = {
        # the next line raises an error so we need to change it
        #'subscription_probability': y_pred,
        'subscription_probability': float(y_pred),
        #'subscription': churn
        'subscription': bool(subscription)
    }
 
    return jsonify(result) 
 
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)