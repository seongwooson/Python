import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline

dataset = pd.read_csv("C:/Users/tjddn/Desktop/pytorch/chap02/data/car_evaluation.csv")
dataset.head()  #dataset.tail(): call last line of data

fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 8
fig_size[1] = 6
plt.rcParams["figure.figsize"] = fig_size
dataset.output.value_counts().plot(kind = 'pie', autopct = '%0.05f',
                                   colors=['lightblue', 'lightgreen', 'orange', 'pink'], explode=(0.05, 0.05, 0.05, 0.05))
#plt.show()
#category 있는거 만들어주고 각 카테고리마다 dataset있는것들을 astype('category')을 통해서 범주특성을 갖는 데이터를 범주형 타입으로 변환
#이후 해당 범주형 타입 데이터들을 넘파이 배열로 변환 후 텐서로 변환함.
categorical_columns = ['price', 'maint', 'doors', 'persons','lug_capacity','safety']

for category in categorical_columns:
    dataset[category] = dataset[category].astype('category')
print("dataset\n", type(dataset))
#cat.codes - 범주형 데이터(단어)를 숫자(넘파이 배열)로 변환
#어떤 클래스가 어떤 숫자로 mapping 되어있는지 확인 어려운 단점
price = dataset['price'].cat.codes.values
maint = dataset['maint'].cat.codes.values
doors = dataset['doors'].cat.codes.values
persons = dataset['persons'].cat.codes.values
lug_capacity = dataset['lug_capacity'].cat.codes.values
safety = dataset['safety'].cat.codes.values

categorical_data = np.stack([price, maint, doors,persons, lug_capacity, safety], 1)
categorical_data = torch.tensor(categorical_data, dtype=torch.int64)
#print("categorical data\n", categorical_data[:10])

#ouputs로 사용할 칼럼 -> 텐서로 변환
#dataset은 이미 astype으로 범주형 타입으로 변환되어있음.
#print("dataset\n", dataset.output)
outputs = pd.get_dummies(dataset.output) #output 칼럼에 대해 가변수 생성
#print("outputs(dummies)\n", outputs)
outputs = outputs.values #가변수 값들 저장
#print("outputs(values)\n", outputs)
outputs = torch.tensor(outputs).flatten()    #1차원 텐서로 변환
#print("categorical_data shape\n", categorical_data.shape)
#print("ouputs shape\n", outputs.shape)

categorical_columns_sizes = [len(dataset[column].cat.categories) for column in categorical_columns]
#print("categorical columns sizes\n", categorical_columns_sizes)
categorical_embedding_sizes = [(col_size, min(50, (col_size+1)//2)) for col_size in categorical_columns_sizes]
#print(categorical_embedding_sizes)

total_records = 1728
test_records = int(total_records * .2)

categorical_train_data = categorical_data[:total_records - test_records] #80%
categorical_test_data = categorical_data[total_records-test_records:total_records] #rest 20%
train_outputs = outputs[:total_records - test_records]
test_outputs = outputs[total_records - test_records:total_records]


###
# model create
###

class Model(nn.Module):
    def __init__(self, embedding_size, output_size, layers, p=0.4):
        super().__init__()
        self.all_embeddings = nn.ModuleList([nn.Embedding(ni, nf) for ni, nf in embedding_size])
        
        self.embedding_dropout = nn.Dropout(p)
        
        all_layers = []
        num_categorical_cols = sum((nf for ni, nf in embedding_size))
        input_size = num_categorical_cols
        
        for i in layers:
            all_layers.append(nn.Linear(input_size, i))
            all_layers.append(nn.ReLU(inplace=True))
            all_layers.append(nn.BatchNorm1d(i))
            all_layers.append(nn.Dropout(p))
            input_size = i
            
        all_layers.append(nn.Linear(layers[-1], output_size))
        self.layers = nn.Sequential(*all_layers)
        
    def forward(self, x_categorical):
        embeddings = []
        for i,e in enumerate(self.all_embeddings):
            embeddings.append(e(x_categorical[:,i].to(self.all_embeddings[i].weight.device)))
        x=torch.cat(embeddings, 1)
        x=self.embedding_dropout(x)
        x=self.layers(x)
        return x

model = Model(categorical_embedding_sizes, 4, [200,100,50], p = 0.4)
#print(model)

#GPU 유무에 따른 device 선정
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

model = model.to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

epochs = 500
aggregated_losses = []
train_outputs = train_outputs.to(device = device, dtype = torch.int64)

for i in range(epochs):
    i += 1  
    y_pred = model(categorical_train_data).to(device=device)
    single_loss = loss_function(y_pred, train_outputs)
    aggregated_losses.append(single_loss)
    
    if i%25 == 1:
        print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')
        
    optimizer.zero_grad()
    single_loss.backward()
    optimizer.step()

print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')

#evaluate model with test datatsets
test_outputs = test_outputs.to(device = device, dtype=torch.int64)
with torch.no_grad():
    y_val = model(categorical_test_data)
    loss = loss_function(y_val, test_outputs)
print(f'Loss: {loss:.8f}')

#모델 예측 확인
print(y_val[:5])

y_val = torch.argmax(y_val, dim= 1)
print(y_val[:5])

#테스트 데이터셋을 이용해 정확도 확인
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
test_outputs_cpu = test_outputs.cpu()
y_val_cpu = y_val.cpu().numpy()
print(confusion_matrix(test_outputs_cpu, y_val_cpu))
print(classification_report(test_outputs_cpu,y_val_cpu))
print(accuracy_score(test_outputs_cpu, y_val_cpu))