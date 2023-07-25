import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
import torchvision.models as models
from torchvision.models import inception_v3

train_data = pd.read_csv('./Desktop/classify-leaves/train.csv')
test_data = pd.read_csv('./Desktop/classify-leaves/test.csv')

leaves_labels = sorted(list(set(train_data['label'])))
n_classes = len(leaves_labels)
class_to_num = dict(zip((leaves_labels),range(n_classes)))

num_to_class = dict(zip(range(n_classes), (leaves_labels)))
img_pth = './Desktop/classify-leaves/'
train_pth = './Desktop/classify-leaves/train.csv'
test_pth = './Desktop/classify-leaves/test.csv'


class LeavesData(Dataset):
    def __init__(self, csv_pth, file_pth, mode, resize = 256) -> None:
        super().__init__()
        self.height = resize
        self.width = resize
        self.file_pth = file_pth
        self.mode = mode

        self.data_info = pd.read_csv(csv_pth, header = None)
        self.len = len(self.data_info.index) -1

        if mode == 'train':
            self.train_img = np.asarray(self.data_info.iloc[1:,0])
            self.train_label = np.asarray(self.data_info.iloc[1:,1])
            self.img_arr = self.train_img
            self.label_arr = self.train_label
            
        else:
            self.test_img = np.asarray(self.data_info.iloc[1:,0])
            self.img_arr = self.test_img

#        print('Finish {} set of {} samples'.format(mode, self.len))
    
    def __getitem__(self, index):
        single_img_name = self.img_arr[index]
        img = Image.open(self.file_pth + single_img_name)
        transform = transforms.Compose([transforms.Resize((299,299)),transforms.ToTensor()])
        img = transform(img)

        if self.mode == 'test':
            return img
        else:
            label = self.label_arr[index]
            label = class_to_num[label]
            return img, label
    
    def __len__(self):
        return self.len

train_dataset = LeavesData(train_pth, img_pth, mode = 'train')
test_dataset = LeavesData(test_pth, img_pth, mode = 'test')

train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                           batch_size = 16,
                                           shuffle = True,
                                           num_workers = 5)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                           batch_size = 16,
                                           shuffle = False,
                                           num_workers = 5)

device = 'cuda' if torch.cuda.is_available() else 'cpu'


model = inception_v3(init_weights=True).to(device)
nf = model.fc.in_features
model.fc = nn.Linear(nf,n_classes)

learning_rate = 0.001
weight_decay = 0.001
EPOCH = 10
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(),lr = learning_rate, weight_decay=weight_decay)
# if __name__ == '__main__':
#     itera = iter(train_loader)
#     img, label = next(itera)
#     print(len(label))
#     for i in range(4):
#         print(label)
model.load_state_dict(torch.load('./Desktop/model_weights.pth', map_location=device))
acc = 0
if __name__ == '__main__':
    for epoch in range(EPOCH):
        model.train()
        train_loss = []
        train_acc = []
        for batch_index, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            output, _ = model(images)
            loss = criterion(output,labels)
            loss.backward()
            optimizer.step()
            if batch_index % 100 == 0:
                print('Train epoch:{}\t loss:{:.5f}'.format(epoch, loss.item()))
            acc = (output.argmax(dim=-1) == labels).float().mean()

            # Record the loss and accuracy.
            train_loss.append(loss.item())
            train_acc.append(acc)
        train_loss = sum(train_loss) / len(train_loss)
        train_accu = sum(train_acc) / len(train_acc)
        print(f"[ Train | {epoch + 1:03d}/{EPOCH:03d} ] loss = {train_loss:.5f}, acc = {train_accu:.5f}")








