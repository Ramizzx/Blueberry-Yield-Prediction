import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


data = pd.read_csv('WildBlueberryPollinationSimulationData.csv')
data = data.drop(columns='Row#')
print(data.head())
X = data.drop(columns = ['yield'])
y = data['yield']

X_train,X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
rf.fit(X_train,y_train)

print(rf.predict(X_test))
import joblib
joblib.dump(rf,'randomforest.joblib')

print(X_test.head())