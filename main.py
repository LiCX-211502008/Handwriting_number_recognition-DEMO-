from multiprocessing.context import set_spawning_popen
from pickletools import uint8
from random import random
from turtle import forward
import numpy as npy
import cv2 as cv
import struct
import time
import torch
from torch import device, nn
from torch.utils.data import DataLoader
from torchaudio import transforms
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import torchvision.transforms
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import draw_number
import torch.onnx as onnx
import torchvision.models as models
import Model

Device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(Device))


def read_img(add) :
    O=open(add,'rb')
    buffer=O.read()
    magic,num,rows,cols=struct.unpack_from('>IIII',buffer,0)
    bits=rows*cols*num
    images=struct.unpack_from('>'+str(bits)+'B',buffer,struct.calcsize('>IIII'))
    O.close
    images=npy.reshape(images,[num,rows*cols])
    return images

def read_labels(add) :
    O=open(add,'rb')
    buffer=O.read()
    magic,num=struct.unpack_from('>II',buffer,0)
    labels=struct.unpack_from('>'+str(num)+'B',buffer,struct.calcsize('>II'))
    O.close
    labels=npy.reshape(labels,[num])
    return labels


class MNIST_dataset(Dataset):
    def __init__(self,Images,Labels) -> None:
        super().__init__()
        self.images = npy.reshape(Images,[npy.size(Images,0),1,28,28])
        #thresh,self.images = cv.threshold(npy.uint8(self.images),120,255,cv.THRESH_BINARY)
        self.images,self.labels = torch.from_numpy(self.images).to(Device),torch.from_numpy(Labels).to(Device)
        self.images,self.labels = self.images.float(),self.labels.float()
        
    def __len__(self):
        return (self.labels).size(0)

    def __getitem__(self, index):
        X = self.images[index,:]
        Y = npy.zeros([10])
        Y[int(self.labels[index])] = 1
        Y = torch.from_numpy(Y).to(Device)
        return X , Y

model_path = input("input model path : ")
if model_path != '':
    #model = models.vgg16() # we do not specify pretrained=True, i.e. do not load default weights
    model = torch.load(model_path)
    model.eval()
else :
    model = Model.CNN().to(Device)

train_images = read_img(input('输入训练集图片地址及名称:'))
train_labels = read_labels(input('输入训练集图片labels地址及名称:'))
test_images = read_img(input('输入测试集图片地址及名称:'))
test_labels = read_labels(input('输入测试集图片labels地址及名称:'))
print(model)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),lr=1e-3,momentum=0.8)
Batch_size = input("batch size = ")
Batch_size = int(Batch_size)
train_dataloader = DataLoader(MNIST_dataset(train_images,train_labels),batch_size = Batch_size,shuffle = True)
test_dataloader = DataLoader(MNIST_dataset(test_images,test_labels),batch_size = Batch_size,shuffle = True)
def train(dataloader,model,loss_fn,optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch , (X , Y) in enumerate(dataloader):
        X,Y = X.to(Device),Y.to(Device)

        pred = model(X)
        loss = loss_fn(pred,Y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            Loss, current = loss.item(), batch * len(X)
            print(f"loss: {Loss:>7f}  [{current:>5d}/{size:>5d}]")
    
def test(dataloader,model,loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss,correct = 0,0
    with torch.no_grad():
        for X,Y in dataloader:
            X,Y = X.to(Device),Y.to(Device)
            pred = model(X)
            test_loss += loss_fn(pred,Y).item()
            #print("\n pred.argmax(0) = ")
            #print(pred.size())
            #print(Y.argmax(1))
            correct += (pred.argmax(1) == Y.argmax(1)).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

epochs = int(input("epochs = "))
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader,model,loss_fn,optimizer)
    test(test_dataloader,model,loss_fn)

print("Done!!")

for t in range(30):
    num = int(random()*len(test_dataloader))
    Show = test_images[num]
    Show = npy.reshape(Show,(28,28))
    Show = Show.astype(npy.uint8)
    Show = cv.resize(Show,None,fx = 12,fy = 12)
    TEST = test_images[num]
    TEST = npy.reshape(TEST,[1,1,28,28])
    TEST = torch.from_numpy(TEST).to(Device)
    TEST = TEST.float()
    cv.imshow("Lable : " + str(test_labels[num]) + "  " + str(model(TEST).argmax(1)) , Show)
    cv.waitKey(0)
    cv.destroyAllWindows()
    

model_path = input("save model path :")
if model_path != '':
    #model = models.vgg16(pretrained=True)
    torch.save(model, model_path)

while(1):
    Input_numpy = draw_number.draw_number()
    Input = torch.from_numpy(npy.reshape(Input_numpy,[1,1,28,28])).float().to(Device)
    pred = model(Input)
    print(pred.argmax(1))
    Show = npy.reshape(Input_numpy,(28,28))
    Show = cv.resize(Show,None,fx = 12,fy = 12)
    cv.imshow(str(pred.argmax(1)),Show)
    cv.waitKey(0)
