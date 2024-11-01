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

df = pd.read_csv('bank+marketing/bank/bank-full.csv', delimiter=';')
df.y = (df.y == 'yes').astype(int)
y = df.y.values

features = ['job', 'duration', 'poutcome']
dicts = df[features].to_dict(orient='records')

dv = DictVectorizer(sparse=False)
X = dv.fit_transform(dicts)

model = LogisticRegression().fit(X, y)

model_file = "model1.bin"
dv_file = "dv.bin"

with open(model_file, 'rb') as m_in:
    model = pickle.load(m_in)

app = Flask('subscription')

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
        # the next line raises an error so we need to change it
        #'churn': churn
        'subscription': bool(subscription)
    }
 
    return jsonify(result) 
 
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)