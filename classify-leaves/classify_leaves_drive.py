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
# Mount google drive
from google.colab import drive
drive.flush_and_unmount()
drive.mount('/content/drive', force_remount=False)


train_data = pd.read_csv('/content/drive/My Drive/train.csv')
test_data = pd.read_csv('/content/drive/My Drive/test.csv')

leaves_labels = sorted(list(set(train_data['label'])))
n_classes = len(leaves_labels)
class_to_num = dict(zip((leaves_labels),range(n_classes)))

num_to_class = dict(zip(range(n_classes), (leaves_labels)))
img_pth = '/content/drive/My Drive/Images/'
train_pth = '/content/drive/My Drive/train.csv'
test_pth = '/content/drive/My Drive/test.csv'


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
                                           drop_last = True)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                           batch_size = 16,
                                           shuffle = False,
                                           drop_last = True)

device = 'cuda'
model = inception_v3(init_weights=True).to(device)
nf = model.fc.in_features
model.fc = nn.Linear(nf,n_classes)
model.load_state_dict(torch.load('/content/drive/My Drive/model_weights.pth', map_location=device))


learning_rate = 0.001
weight_decay = 0.001
EPOCH = 30
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(),lr = learning_rate, weight_decay=weight_decay)

acc = 0
loss_ls = []
acc_ls = []
if __name__ == '__main__':
    for epoch in range(EPOCH):
        model.train()
        model.to(device)
        train_loss = []
        train_acc = []
        for batch_index, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            output, _ = model(images)
            loss = criterion(output,labels)
            loss.backward()
		        #loss_ls.append(loss.item())
            loss_ls.append(loss.item())
            optimizer.step()
            if batch_index % 100 == 0:
                print('Train epoch:{}\t loss:{:.5f}'.format(epoch, loss.item()))
            acc = (output.argmax(dim=-1) == labels).float().mean()
            acc_ls.append(acc.item())


            # Record the loss and accuracy.
            train_loss.append(loss.item())
            train_acc.append(acc)
        train_loss = sum(train_loss) / len(train_loss)
        train_accu = sum(train_acc) / len(train_acc)
        print(f"[ Train | {epoch + 1:03d}/{EPOCH:03d} ] loss = {train_loss:.5f}, acc = {train_accu:.5f}")

    torch.save(model.state_dict(), 'model_weights.pth')
    plt.plot(loss_ls)
    plt.title('Train Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Train Loss')
    plt.show()

    plt.plot(train_acc)
    plt.title('Train ACC')
    plt.xlabel('Epochs')
    plt.ylabel('Train ACC')
    plt.show()