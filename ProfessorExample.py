import pandas as pd
import numpy as np
from sklearn import svm

Path = 'Final_Project/Dataset.xlsx'
df = pd.read_excel(Path)
Home = []
for i in range(df.shape[0]):
    if df.iloc[:, 1][i][4:6]=='vs':
        Home.append(1)
    if df.iloc[:,1][i][4:6]=='@ ':
        Home.append(0)
df['home'] = np.array(Home)
df.loc[df['W/L'] == 'W', 'W/L'] = 1
df.loc[df['W/L'] == 'L', 'W/L'] = 0
df.loc[df['FT%']=='-','FT%']=1


Features = ['W/L','PTS', 'FGM', 'FGA',
       'FG%', '3PM', '3PA', '3P%', 'FTM', 'FTA', 'FT%', 'OREB', 'DREB', 'REB',
       'AST', 'STL', 'BLK', 'TOV', 'PF', '+/-']
for i in Features:
    df[i] = df[i].astype(float)

X_all = []
Y_all = []
def Weighting_fun(L,alpha=0.5):
    Res = [alpha ** i for i in np.arange(L,0,-1)]
    Res /= np.sum(Res)
    return Res


for i in np.arange(300,df.shape[0]):
    Team_1 = df.iloc[i,0]
    Team_2 = df.iloc[i,1][-3::]
    Data_Previous_1 = (df.iloc[0:i,:][df.iloc[0:i,0]==Team_1])
    Data_Previous_2 = (df.iloc[0:i,:][df.iloc[0:i,0]==Team_2])
    Data_Previous_1_home = np.array(Data_Previous_1[Data_Previous_1['home'] == df.iloc[i, -1]][Features])
    Data_Previous_2_home = np.array(Data_Previous_2[Data_Previous_2['home'] == (1-df.iloc[i, -1])][Features])
    W1 = Weighting_fun(Data_Previous_1_home.shape[0],0.4).reshape(-1,1)
    W2 = Weighting_fun(Data_Previous_2_home.shape[0],0.4).reshape(-1,1)
    Data_1 = np.array(Data_Previous_1_home[:,3::] * W1).mean(axis=0)
    Data_2 = np.array(Data_Previous_2_home[:,3::] * W2).mean(axis=0)
    Diff = (Data_1 - Data_2).tolist() + [df.iloc[i,-1]]
    X_all.append(Diff)
    Y_all.append(df.iloc[i]['W/L'])

X = np.array(X_all)
y = np.array(Y_all)
# Load a dataset (using iris dataset as example)
def z_score_normalize(arr):
    mean_val = arr.mean(axis=0)
    std_val = arr.std(axis=0)
    return (arr - mean_val) / std_val

X_train, X_test, y_train, y_test = X[0:1500,:], X[1500::,:], y[0:1500],y[1500::]
model = svm.SVC()
model.fit(X_train, y_train)
# Predict using the SVM model
predictions = model.predict(X_test)
accuracy = model.score(X_test, y_test)
train_accuracy = model.score(X_train, y_train)
print("Accuracy of SVM:", accuracy,train_accuracy)


