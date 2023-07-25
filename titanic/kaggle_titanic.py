import pandas as pd
import numpy as np
import torch
train_data = pd.read_csv('./Desktop/kaggle/titanic/train.csv')
test_data = pd.read_csv('./Desktop/kaggle/titanic/test.csv')

train_data.to_csv('./Desktop/train.csv',index=False, encoding='utf-8')    

data = pd.concat((train_data.iloc[:,2:], test_data.iloc[:,1:]))
for i in range(len(data)):
    if data.iloc[i,2] == 'male':
        data.iloc[i,2] = 1
    elif data.iloc[i,2] == 'female':
        data.iloc[i,2] = 2

    
    if isinstance(data.iloc[i,8],str):
        letter = data.iloc[i,8][0]
        data.iloc[i,8] = ord(letter)
    
    if isinstance(data.iloc[i,9],str):
        letter = data.iloc[i,9]
        data.iloc[i,9] = ord(letter)
data = data.fillna(0)
n_data = data.dtypes[data.dtypes != 'object'].index

data[n_data] = data[n_data].apply(lambda x: (x-x.mean())/(x.std()))

n_train = train_data.shape[0]
train = torch.tensor(data[n_data][:n_train].values,dtype=torch.float32)
test = torch.tensor(data[n_data][n_train:].values, dtype=torch.float32)

label = torch.tensor(train_data.Survived.values.reshape(-1,1),dtype = torch.float32)
in_feature = train.shape[1]

def net():
    net = torch.nn.Sequential(
            torch.nn.Linear(in_feature, 300),
            torch.nn.ReLU(),
            torch.nn.Linear(300,350),
            torch.nn.ReLU(),
            torch.nn.Linear(350,1),
            )
    return net

def result(input):
    result = []
    for i in input:
        if i >= 0.5:
            result.append(1)
        else:
            result.append(0)
    return result

def acc(label,result):
    n = 0
    for i, j in zip(label,result):
        if i == j:
            n += 1
    return n/len(result)

EPOCH = 350
lr = 0.1
decay = 0.1
 
net = net()
loss = torch.nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=decay)
for epoch in range(EPOCH):
    optimizer.zero_grad()
    pred_label = net(train)
    l = loss(pred_label, label)
    l.backward()
    optimizer.step()

re = result(pred_label)
a = acc(label, re)

print(a)

GBCpreData_y=result(net(test))


#导出预测结果
GBCpreResultDf=pd.DataFrame()
GBCpreResultDf['PassengerId']=test_data['PassengerId']
GBCpreResultDf['Survived']=GBCpreData_y
GBCpreResultDf
#将预测结果导出为csv文件
#GBCpreResultDf.to_csv('./Desktop/SurviveTest.csv',index=False)

    



