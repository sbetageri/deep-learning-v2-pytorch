import numpy as np
from glob import glob
import torchvision
from tqdm import tqdm

# load filenames for human and dog images
human_files = np.array(glob("lfw/*/*"))
dog_files = np.array(glob("dogImages/*/*/*"))

# print number of images in each dataset
print('There are %d total human images.' % len(human_files))
print('There are %d total dog images.' % len(dog_files))

import torch
import torchvision.models as models

# define VGG16 model
VGG16 = models.vgg16(pretrained=True)

# check if CUDA is available
use_cuda = torch.cuda.is_available()

# move model to GPU if CUDA is available
    
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

VGG16 = VGG16.to(device)

import torchvision.transforms as transforms

def VGG16_predict(img_path):
    '''
    Use pre-trained VGG-16 model to obtain index corresponding to 
    predicted ImageNet class for image at specified path
    
    Args:
        img_path: path to an image
        
    Returns:
        Index corresponding to VGG-16 model's prediction
    '''
    
    ## TODO: Complete the function.
    ## Load and pre-process an image from the given img_path
    ## Return the *index* of the predicted class for that image
    with torch.no_grad():
        img = Image.open(img_path)
        trans1 = transforms.Resize((224, 224))
        trans2 = transforms.ToTensor()
        img = trans1(img)
        img = trans2(img)
        img = img.view(1, 3, 224, 224)
        img = img.to(device)
        out = VGG16(img)
        index = torch.topk(out, 1)[1].item()
    
    return index # predicted class index

import os
from torchvision import datasets

### TODO: Write data loaders for training, validation, and test sets
## Specify appropriate transforms, and batch_sizes

transformations = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
])

train_dataset = torchvision.datasets.ImageFolder('./dogImages/train/', transform=transformations)
val_dataset = datasets.ImageFolder('./dogImages/valid/', transform=transformations)
test_dataset = datasets.ImageFolder('./dogImages/test/', transform=transformations)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, shuffle=True, batch_size=32)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, shuffle=True, batch_size=32)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, shuffle=True, batch_size=32)

loaders_scratch = {
    'train':train_loader,
    'valid':val_loader,
    'test':test_loader
}

import torchvision

import torch.nn as nn
import torch.nn.functional as F

# define the CNN architecture
class Net(nn.Module):
    ### TODO: choose an architecture, and complete the class
    def __init__(self):
        super(Net, self).__init__()
        ## Define layers of a CNN
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 2, stride=2, padding=1),
            nn.ReLU()
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256*4*4, 1024),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, 133)
        )
    
    def forward(self, x):
        ## Define forward behavior
        x = self.features(x)
        x = x.view(x.size(0), 256*4*4)
        x = self.classifier(x)
#         x = F.softmax(x, dim=1)
        return x
    

#-#-# You so NOT have to modify the code below this line. #-#-#

# instantiate the CNN
model_scratch = Net()

# move tensors to GPU if CUDA is available
model_scratch = model_scratch.to(device)


import torch.optim as optim

### TODO: select loss function
criterion_scratch = nn.CrossEntropyLoss()

### TODO: select optimizer
optimizer_scratch = optim.SGD(model_scratch.parameters(), lr=0.0001)

def train(n_epochs, loaders, model, optimizer, criterion, use_cuda, save_path):
    """returns trained model"""
    # initialize tracker for minimum validation loss
    valid_loss_min = np.Inf 
    
    for epoch in range(1, n_epochs+1):
        # initialize variables to monitor training and validation loss
        train_loss = 0.0
        valid_loss = 0.0
        train_acc = 0
        val_acc = 0
        
        ###################
        # train the model #
        ###################
        model.train()
        for data, target in tqdm(loaders['train']):
            # move to GPU
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            out = model(data)
#             print(target)
#             print(out)
            out_idx = torch.topk(out, 1)[1]
            out_idx = out_idx.view(out_idx.size()[0])
            train_acc += torch.sum(out_idx == target).item()
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss = train_loss / (len(loaders['train']) * 32)
        train_acc = train_acc / (len(loaders['train']) * 32)
        train_acc *= 100
#             train_loss = ((train_loss * batch_idx) + loss.detach().item()) / (batch_idx + 1)
            ## find the loss and update the model parameters accordingly
            ## record the average training loss, using something like
            ## train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
            
        ######################    
        # validate the model #
        ######################
        model.eval()
        for data, target in tqdm(loaders['valid']):
            # move to GPU
            data, target = data.to(device), target.to(device)
            ## update the average validation loss
            out = model(data)
            loss = criterion(out, target)
            valid_loss += loss.item()
            out_idx = torch.topk(out, 1)[1]
            out_idx = out_idx.view(out_idx.size()[0])
            val_acc += torch.sum(out_idx == target).item()
        valid_loss = valid_loss / (len(loaders['valid']) * 32)
        val_acc = val_acc / (len(loaders['valid']) * 32)
        val_acc *= 100
#             valid_loss = ((valid_loss * batch_idx) + loss.detach().item()) / (batch_idx + 1)

            
        # print training/validation statistics 
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, 
            train_loss,
            valid_loss
            ))
        print('Epoch : {} \tTraininc Acc: {:.6f} \tValidation Acc : {:.6f}'.format(
            epoch,
            train_acc,
            val_acc
        ))
        
        ## TODO: save the model if validation loss has decreased
        if valid_loss < valid_loss_min:
            valid_loss_min = valid_loss
            torch.save(model.state_dict(), './' + save_path)
            
    # return trained model
    return model


# train the model
model_scratch = train(100, loaders_scratch, model_scratch, optimizer_scratch, 
                      criterion_scratch, use_cuda, 'model_scratch.pt')

# load the model that got the best validation accuracy
model_scratch.load_state_dict(torch.load('model_scratch.pt'))

def test(loaders, model, criterion, use_cuda):

    # monitor test loss and accuracy
    test_loss = 0.
    correct = 0.
    total = 0.

    model.eval()
    for batch_idx, (data, target) in enumerate(loaders['test']):
        # move to GPU
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the loss
        loss = criterion(output, target)
        # update average test loss 
        test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss.data - test_loss))
        # convert output probabilities to predicted class
        pred = output.data.max(1, keepdim=True)[1]
        # compare predictions to true label
        correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
        total += data.size(0)
            
    print('Test Loss: {:.6f}\n'.format(test_loss))

    print('\nTest Accuracy: %2d%% (%2d/%2d)' % (
        100. * correct / total, correct, total))

# call test function    
test(loaders_scratch, model_scratch, criterion_scratch, use_cuda)