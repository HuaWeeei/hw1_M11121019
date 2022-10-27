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

df = pd.read_csv("train.csv")
dt = pd.read_csv("test.csv")
X = df.iloc[:,0:12]
op = X.columns
y = df['Class']

labelencoder = LabelEncoder() #類別化
data_le = pd.DataFrame(X)
data_le['Auction_ID']= labelencoder.fit_transform(data_le['Auction_ID'])
data_le['Bidder_ID']= labelencoder.fit_transform(data_le['Bidder_ID'])
X = data_le
X_s = StandardScaler()     #標準化  
X = X_s.fit_transform(X)
train_X, test_X, train_y, test_y = train_test_split(X,y,test_size=0.2)
dectree_grid = DecisionTreeClassifier(criterion='gini')
cv_params = {
            # 'criterion':['gini','entropy'],
                 'max_depth':np.linspace(10,100,10,dtype=int) , 'splitter':['random','best'] }
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
graph.write_pdf('tree.pdf')
tree.plot_tree(dectree)
#----------------------測試集
Xt = dt.iloc[:,0:12]
data_te = pd.DataFrame(Xt)
data_te['Auction_ID']= labelencoder.fit_transform(data_te['Auction_ID'])
data_te['Bidder_ID']= labelencoder.fit_transform(data_te['Bidder_ID'])
Xt = data_te
Xt = X_s.fit_transform(Xt)
result = dectree.predict(Xt)
yt = dt['Class']
cla = classification_report(yt,result)
print('test：',cla)
result=pd.DataFrame(result)
result.to_csv('DM1_ans.csv',index = False)
yt.to_csv('DM1_ori.csv',index = False)
