
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
from torchvision.models import resnet18

#train_data = pd.read_csv('./Desktop/dc/train.csv')
#test_data = pd.read_csv('./Desktop/dc/test.csv')

train_pth = './Desktop/dc/train/'
test_pth = './Desktop/dc/test/'
length = len(os.listdir(train_pth))



class Data(Dataset):
    def __init__(self,img_pth, mode):
        self.img_pth = img_pth
        self.mode = mode
        self.len = len(os.listdir(img_pth))
        if mode == 'train':
            self.train_label = np.asarray(os.listdir(img_pth))
            self.train_img = np.asarray(os.listdir(img_pth))
            self.img_arr = self.train_img
            self.label_arr = self.train_label
            
        else:
            self.test_label = np.asarray(os.listdir(img_pth))
            self.test_img = np.asarray(os.listdir(img_pth))
            self.img_arr = self.test_img
    
    def __getitem__(self, index):
        img_name = self.img_arr[index]
        img = Image.open(self.img_pth+img_name)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
        trans = transforms.Compose([transforms.ToTensor(),
                                            transforms.Resize((256, 256)),
                                            normalize])
        img = trans(img)
        if self.mode == 'train':
            label = img_name[:3]
            if label == 'cat':
                label = 0
            else:
                label = 1
            return img, label
        else:
            return img
    
    def __len__(self):
        return self.len

train_dataset = Data(train_pth, mode= 'train')
test_dataset = Data(test_pth, mode = 'test')

train_loader = DataLoader(dataset= train_dataset,
                          batch_size = 16,
                          shuffle = True,
                          drop_last = True)

test_loader = DataLoader(dataset= test_dataset,
                         batch_size = 16,
                         shuffle= False,
                         drop_last= True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# detailiter = iter(train_loader)
# img, label = next(detailiter)
# fig = plt.figure(figsize=(20,12))
# c = 4
# r = 2
# def im_convert(tensor):
#     """ 展示数据"""
    
#     image = tensor.to("cpu").clone().detach()
#     image = image.numpy().squeeze()
#     image = image.transpose(1,2,0)
#     image = image.clip(0, 1)

#     return image
# for i in range(8):
#     ax = fig.add_subplot(r,c,i+1,xticks=[],yticks=[])
#     ax.set_title(label[i][:3])
#     plt.imshow(im_convert(img[i]))
# plt.show()

model = resnet18(weights = True)
model.to(device)
num_features = model.fc.in_features
num_layers_to_remove = 2  # Number of layers to remove

#for _ in range(num_layers_to_remove):

#    model = torch.nn.Sequential(*list(model.children()))[:-1]

num_classes = 2  # Replace with the number of classes in your classification problem
model.fc=torch.nn.Linear(num_features,num_classes)


learning_rate = 0.001
weight_decay = 0.001
EPOCH = 20
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(),lr = learning_rate,weight_decay=weight_decay)
acc = 0

if __name__ == '__main__':
    loss_ls = []
    acc_ls = []
    for epoch in range(EPOCH):
        model.train()
        train_loss = []
        train_acc = []
        for batch_index, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output,labels)
            loss.backward()
            optimizer.step()
            loss_ls.append(loss.item())
            if batch_index % 300 == 0:
                print('Train epoch:{}\t loss:{:.5f}'.format(epoch, loss.item()))
            acc = (output.argmax(dim=-1) == labels).float().mean()

            # Record the loss and accuracy.
            train_loss.append(loss.item())
            train_acc.append(acc.item())
            print(train_acc)
            
        train_loss = sum(train_loss) / len(train_loss)
        train_accu = sum(train_acc) / len(train_acc)
        acc_ls.append(train_accu)
    
        print(f"[ Train | {epoch + 1:03d}/{EPOCH:03d} ] loss = {train_loss:.5f}, acc = {train_accu:.5f}")
    
    torch.save(model.state_dict(), 'model_weight.pth')
    plt.title('Train Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(loss_ls)
    plt.show()

    plt.title('Train Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.plot(acc_ls)
    plt.show()




        # with torch.no_grad():
        #     model.eval()
        #     for batch_index, img in enumerate(test_loader):
        #         img = img.to(device)

        #         output = model(img)
        #         _, prediciton = torch.max(output.data, dim=-1)
                    
'''''

for i in ls:
    if i == 0:
        lst.append('cat')
    else:
        lst.append('dog')
import pandas as pd
GBCpreResultDf=pd.DataFrame()
GBCpreResultDf['id']=range(1,len(lst)+1)
GBCpreResultDf['label']=ls
GBCpreResultDf.to_csv('./Desktop/dogvscat.csv',index=False)
'''