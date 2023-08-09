import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.impute import SimpleImputer
from sklearn_features.transformers import DataFrameSelector
import os

FILE_PATH = os.path.join(os.getcwd(), 'housing.csv')
data = pd.read_csv(FILE_PATH)

data['ocean_proximity'].replace('<1H OCEAN','1H OCEAN',inplace=True)

data['rooms_per_household']=data['total_rooms']/data['households']
data['beadroom_per_rooms']=data['total_bedrooms']/data['total_rooms']
data['population_per_household'] = data['population'] / data['households']

x=data.drop('median_house_value',axis=1)
y=data['median_house_value']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.15,shuffle=True,random_state=42)
x_train.shape,x_test.shape,y_train.shape,y_test.shape

#split the data types 
num_cols=[col for col in x_train.columns if x_train[col].dtype == 'float64']
cat_cols=[col for col in x_train.columns if x_train[col].dtype != 'float64']

#impute and fill nulls
imputer=SimpleImputer(strategy='median')
x_train_full=imputer.fit_transform(x_train[num_cols])
x_test_full=imputer.transform(x_test[num_cols])


#Building a pipline for numerical variables
num_pipline=Pipeline(steps=[
    ('selector',DataFrameSelector(num_cols)),
    ('imputer',imputer),
    ('scaler',StandardScaler())

])

#deal with (num_pipline) as an instance -- fit and transform
x_train_num=num_pipline.fit_transform(x_train[num_cols])
x_test_num=num_pipline.transform(x_test[num_cols])
cat_pipline=Pipeline(steps=[('selector',DataFrameSelector(cat_cols)),('imputer',SimpleImputer(strategy='constant',fill_value='missing')
                             ),('OHE',OneHotEncoder(sparse=False))])
x_train_cat=cat_pipline.fit_transform(x_train[cat_cols])
x_test_cat=cat_pipline.transform(x_test[cat_cols])

#concatenate the pipline
total_pipline=FeatureUnion(transformer_list=[('nim_pipline',num_pipline),('cat_pipline',cat_pipline)]
                           )
x_train_final=total_pipline.fit_transform(x_train)

def pre_preporcess(x_new):
    return total_pipline.transform(x_new)
