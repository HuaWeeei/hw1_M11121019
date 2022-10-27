# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 15:21:22 2022

@author: bbill
"""


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report,confusion_matrix
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import numpy as np

df = pd.read_csv("train_adult.csv")
dt = pd.read_csv("test_adult.csv")
X = df.iloc[:,0:14]
op = X.columns
y = df['target']
X = X.drop("fnlwgt",axis=1)
X = X.drop("capital-gain",axis=1)
X = X.drop("capital-loss",axis=1)
labelencoder = LabelEncoder() #類別化
data_le = pd.DataFrame(X)
data_le['workclass']= labelencoder.fit_transform(data_le['workclass'])
data_le['education']= labelencoder.fit_transform(data_le['education'])
data_le['marital-status']= labelencoder.fit_transform(data_le['marital-status'])
data_le['occupation']= labelencoder.fit_transform(data_le['occupation'])
data_le['relationship']= labelencoder.fit_transform(data_le['relationship'])
data_le['race']= labelencoder.fit_transform(data_le['race'])
data_le['sex']= labelencoder.fit_transform(data_le['sex'])
data_le['native-country']= labelencoder.fit_transform(data_le['native-country'])
X = data_le
X_s = StandardScaler()     #標準化  
X = X_s.fit_transform(X)
train_X, test_X, train_y, test_y = train_test_split(X,y,test_size=0.2)
dectree_grid = DecisionTreeClassifier(criterion='entropy')
cv_params = {
            # 'criterion':['gini','entropy'],
                 'max_depth':np.linspace(1,10,10,dtype=int) , 'splitter':['random','best'] }
from sklearn.model_selection import GridSearchCV
gs_m = GridSearchCV(dectree_grid , cv_params, verbose=2
                   ,refit=True, n_jobs=1 ,cv=10)
gs_m.fit(train_X,train_y)
dectree = gs_m.best_estimator_
print(f"最佳準確率: {gs_m.best_score_}，最佳參數組合：{gs_m.best_params_}")
predictions = dectree.predict(test_X)
# accuracy = metrics.accuracy_score(test_y, predictions)
vcla = classification_report(test_y,predictions)
print('vali',vcla)
import pydotplus
from sklearn import tree
dot_data = tree.export_graphviz(dectree, out_file=None)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf('entropy14.pdf')
#----------------------測試集
Xt = dt.iloc[:,0:14]
Xt = Xt.drop("fnlwgt",axis=1)
Xt = Xt.drop("capital-gain",axis=1)
Xt = Xt.drop("capital-loss",axis=1)
data_tle = pd.DataFrame(Xt)
data_tle['workclass']= labelencoder.fit_transform(data_tle['workclass'])
data_tle['education']= labelencoder.fit_transform(data_tle['education'])
data_tle['marital-status']= labelencoder.fit_transform(data_tle['marital-status'])
data_tle['occupation']= labelencoder.fit_transform(data_tle['occupation'])
data_tle['relationship']= labelencoder.fit_transform(data_tle['relationship'])
data_tle['race']= labelencoder.fit_transform(data_tle['race'])
data_tle['sex']= labelencoder.fit_transform(data_tle['sex'])
data_tle['native-country']= labelencoder.fit_transform(data_tle['native-country'])
Xt = data_tle
Xt = X_s.fit_transform(Xt)
result = dectree.predict(Xt)
yt = dt['target']
cla = classification_report(yt,result)
print('test：',cla)
result=pd.DataFrame(result)
# result.to_csv('DM1_ans.csv',index = False)
# yt.to_csv('DM1_ori.csv',index = False)
