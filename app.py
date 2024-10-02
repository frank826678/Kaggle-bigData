from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import numpy as np
import os
import gc
from glob import glob
from pathlib import Path
from datetime import datetime
import polars as pl
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
import lightgbm as lgb
import plotly.express as px
import plotly.io as pio  # 用來將圖表轉為HTML

app = Flask(__name__)

#class VotingModel(BaseEstimator, ClassifierMixin):
#    def __init__(self, estimators):
#        super().__init__()
#        self.estimators = estimators
#        
#    def fit(self, X, y=None):
#        return self
#    
#    def predict(self, X):
#        y_preds = [estimator.predict(X) for estimator in self.estimators]
#        return np.mean(y_preds, axis=0)
#    
#    def predict_proba(self, X):
#        y_preds = [estimator.predict_proba(X) for estimator in self.estimators]
#        return np.mean(y_preds, axis=0)

class VotingModel(BaseEstimator, RegressorMixin):
    def __init__(self, estimators):
        super().__init__()
        self.estimators = estimators
        
    def fit(self, X, y=None):
        return self
    
    def predict(self, X):
        y_preds = [estimator.predict(X) for estimator in self.estimators]
        return np.mean(y_preds, axis=0)
     
    def predict_proba(self, X):      
        # lgb
        lgb_X=X[lgb_cols]
        nums=lgb_X.select_dtypes(exclude='category').columns
        lgb_X[nums] = lgb_X[nums].fillna(0)
        y_preds = [estimator.predict_proba(X) for estimator in self.estimators[:5]]
        del lgb_X
        gc.collect()
        
        # cat 
        #X[cabcat_cols] = X[cabcat_cols].astype(str)
        #y_preds = [estimator.predict_proba(X[cab_cols]) for estimator in self.estimators[-5:]]
        
        return np.mean(y_preds, axis=0)

# 路徑設定
PARQUET_PATH = 'database/df_train_100.parquet'
MODEL_PATH  = 'model/lgb_models.joblib'
MODEL_PATH2 = 'model/cat_models.joblib'
#MODEL_PATH = 'model/lgb_models_nan.joblib'

df = pd.read_parquet(PARQUET_PATH, engine='pyarrow')

#df = pd.read_csv('database/df_train_100.csv', encoding='utf-8')


category_columns = df.select_dtypes('category').columns
df[category_columns] = df[category_columns].astype('object')

object_columns = df.select_dtypes('object').columns
df[object_columns] = df[object_columns].fillna('nan')
df[object_columns] = df[object_columns].astype('category')

object_columns = df.select_dtypes('bool').columns
df[object_columns] = df[object_columns].astype('category')


# 加載模型
lgb_model = joblib.load(MODEL_PATH)
lgb_notebook_info = joblib.load('model/notebook_info.joblib')
lgb_cols = lgb_notebook_info['cols']
lgbcat_cols = lgb_notebook_info['cat_cols']

#cat_model = joblib.load(MODEL_PATH2)
#cat_notebook_info = joblib.load('model/cat_notebook_info.joblib')
#cab_cols = cat_notebook_info['cols']
#cabcat_cols = cat_notebook_info['cat_cols']

model = VotingModel(lgb_model)
#model = VotingModel(model)


@app.route('/')
def index():
    case_ids = df['case_id'].tolist()
    return render_template('index.html', case_ids=case_ids)
    

@app.route('/get_data', methods=['POST'])
def get_data():
    try:
        case_id = int(request.form.get('case_id'))
        print("case_id :", case_id)

        if case_id not in df['case_id'].values:
            return jsonify({'error': 'Invalid case_id'}), 400
        
        user_data = df[df['case_id'] == case_id].to_dict(orient='records')[0]

        gender = user_data.get('max_sex_738L', 'N/A')
        income_money = user_data.get('maininc_215A', 'N/A')
        age = round(abs(user_data.get('max_birth_259D', 'N/A')) / 365,0)
        loan_rejected = user_data.get('numrejects9m_859L', 0)
        loan_amount = user_data.get('credamount_770A', 'N/A')
        family = user_data.get('max_familystate_726L', 'N/A')
        relation_person = user_data.get('last_relationshiptoclient_642T', 'N/A')
        occupation = user_data.get('max_empl_industry_691L', 'N/A')
        work_year = user_data.get('max_empl_employedtotal_800L', 'N/A')
        income_type = user_data.get('max_incometype_1044T', 'N/A')
        pmtnum = user_data.get('pmtnum_254L', 'N/A')
        past_2year = user_data.get('avgdpdtolclosure24_3658938P', 0)
        days180 = user_data.get('days180_256L', 0)
        eir = user_data.get('eir_270L', 'N/A')

        #print("gender : ", gender)
        #print("income_money : ", income_money)
        #print("age : ", age)
        #print("loan_rejected : ", loan_rejected)
        #print("loan_amount : ", loan_amount)
        #print("family : ", family)
        #print("relation_person : ", relation_person)
        #print("occupation : ", occupation)
        #print("work_year : ", work_year)
        #print("income_type : ", income_type)


        # 先移除不需要的列，然後移除索引
        features = df[df['case_id'] == case_id].drop(columns=['case_id', 'target', 'WEEK_NUM']).reset_index(drop=True)
        
        # 確認features的形狀
        #print("features after drop columns :", features.shape)
        #print("features columns : ", features)
        #print("type of features : ", type(features))
        
        probability = model.predict_proba(features)[0][1]  # 獲取預測結果的概率
        print("prob with model : ", probability)

        return jsonify({
            'user_data': {
                'max_sex_738L': gender,
                'maininc_215A': income_money,
                'max_birth_259D': age,
                'numrejects9m_859L' : loan_rejected,
                'credamount_770A': loan_amount,
                'max_familystate_726L': family,
                'last_relationshiptoclient_642T': relation_person,
                'max_empl_industry_691L': occupation,
                'max_empl_employedtotal_800L' : work_year,
                'max_incometype_1044T': income_type,
                'pmtnum_254L': pmtnum,
                'avgdpdtolclosure24_3658938P': past_2year,
                'days180_256L': days180,
                'eir_270L': eir,
            },
            'probability': f"{probability}"
            #'probability': f"{probability:.2%}"
        })
    except Exception as e:
        print("Error:", str(e))
        return jsonify({'error': 'Internal Server Error'}), 500



#@app.route('/index2.html')
#def index2():
#    return render_template('index2.html')  

@app.route('/index3.html', methods=['POST'])
def index3():
    # # 讀取已生成的 HTML 文件並返回
    with open('static/images/probability_distribution.html', 'r', encoding='utf-8') as f:
        graph_html = f.read()
    return render_template('index3.html', graph_html=graph_html)


if __name__ == '__main__':
    app.run(debug=True)