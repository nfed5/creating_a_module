
# coding: utf-8

# In[27]:


import numpy as np
import pandas as pd
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin


class used_car_model():
    
    def __init__(self, model_file):
        # read the 'model' and 'scaler' files which were saved
        with open('model','rb') as model_file:
            self.reg = pickle.load(model_file)
            #self.data = None
    
    #take the new data and preprocess it
    def load_and_clean_data(self, data_file):
        #import the data
        raw_data = pd.read_csv(data_file)
        #make a copy
        df = raw_data.copy()
        #drop variables
        df = df.drop(['Model'], axis=1)
        df = df.drop(['Registration'], axis=1)
        df = df.drop(['Year'], axis=1)
        #drop missing values
        df = df.dropna(axis=0)
        #drop outliers
        q = df['Price'].quantile(0.99)
        df = df[df['Price']<q]
        q2 = df['Mileage'].quantile(0.99)
        df = df[df['Mileage']<q2]
        df = df[df['EngineV']<6.5]
        df = df.reset_index(drop=True)
        #Log transformation
        log_price = np.log(df['Price'])
        df['log_price'] = log_price
        #make another copy for analysis in Tableau
        self.viz_data = df.copy()
        df = df.drop(['Price'], axis=1)
        
        #Scale data
        features_to_scale = df[['log_price','Mileage','EngineV']]
        scaler = StandardScaler()
        scaler.fit(features_to_scale)
        features_scaled = scaler.transform(features_to_scale)
        
        new_df = pd.DataFrame(data=features_scaled, columns=[['Log Price','mileage','engine volume']])
        combined_df = pd.concat([new_df,df],axis=1)
        combined_df = combined_df.drop(['log_price'],axis=1)
        combined_df = combined_df.drop(['Mileage'],axis=1)
        combined_df = combined_df.drop(['EngineV'],axis=1)
        
        #create dummies
        data_with_dummies = pd.get_dummies(combined_df, drop_first=True)
        
        #use this if you want to call 'preprocessed data'
        self.preprocessed_data = data_with_dummies.copy()
        
        features = self.preprocessed_data.drop(self.preprocessed_data.columns[0],axis=1)
        #this is an np.array of all the x variables that you will use on reg.predict to make predictions
        self.data = features.to_numpy()
        
    
    
    #take the preprocessed data and return a dataframe with predictions
    def predict_output(self):
        
        #self.preprocessed_data['Prediction'] = np.exp(self.reg.predict(self.data))
        self.viz_data['Prediction'] = np.exp(self.reg.predict(self.data))
        
        return self.viz_data

