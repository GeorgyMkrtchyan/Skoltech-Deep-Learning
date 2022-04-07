# Don't erase the template code, except "Your code here" comments.

import subprocess
import sys

# List any extra packages you need here
PACKAGES_TO_INSTALL = ["gdown==4.4.0",]
subprocess.check_call([sys.executable, "-m", "pip", "install"] + PACKAGES_TO_INSTALL)

import torch
# Your code here...
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
import os
from tqdm import tqdm

def get_dataloader(path, kind):
    """
    Return dataloader for a `kind` split of Tiny ImageNet.
    If `kind` is 'val', the dataloader should be deterministic.

    path:
        `str`
        Path to the dataset root - a directory which contains 'train' and 'val' folders.
    kind:
        `str`
        'train' or 'val'

    return:
    dataloader:
        `torch.utils.data.DataLoader` or an object with equivalent interface
        For each batch, should yield a tuple `(preprocessed_images, labels)` where
        `preprocessed_images` is a proper input for `predict()` and `labels` is a
        `torch.int64` tensor of shape `(batch_size,)` with ground truth class labels.
    """
    # Your code here
    if kind == 'train':
        tr_path=os.path.join(path,kind)
        transform_tr1 = transforms.Compose([transforms.RandomHorizontalFlip(),
                            transforms.CenterCrop(56),
                            transforms.Resize(64),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406]
                                                 ,std=[0.229, 0.224, 0.225])])

        transform_tr2 = transforms.Compose([torchvision.transforms.RandomRotation(10),
                                  torchvision.transforms.RandomHorizontalFlip(),
                                  torchvision.transforms.RandomVerticalFlip(),
                                   torchvision.transforms.GaussianBlur(5, sigma=(0.1, 2.0)),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=[0.485, 0.456, 0.406]
                                                        ,std=[0.229, 0.224, 0.225])])

        train = ImageFolder(tr_path,transform=transform_tr1)
        train1=ImageFolder(tr_path,transform=transform_tr2)
        train_big = torch.utils.data.ConcatDataset([train,train1])
        batch_size=256
        dataloader=torch.utils.data.DataLoader(train_big, batch_size=batch_size, 
                              shuffle=True, drop_last=False,num_workers=1)
       
    if kind == 'val':
        val_path=os.path.join(path,kind)
        transform_val = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                 std=[0.229, 0.224, 0.225])])
        val = ImageFolder(val_path,transform=transform_val)
        batch_size=256
        dataloader=torch.utils.data.DataLoader(val, batch_size=batch_size,
                            shuffle=False, drop_last=False, num_workers=1)
    return dataloader



        

def get_model():
    """
    Create neural net object, initialize it with raw weights, upload it to GPU.

    return:
    model:
        `torch.nn.Module`
    """
    # Your code here
    model = torchvision.models.densenet161(pretrained=False)
    model.classifier = torch.nn.Linear(2208, 200)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    return model

def get_optimizer(model):
    """
    Create an optimizer object for `model`, tuned for `train_on_tinyimagenet()`.

    return:
    optimizer:
        `torch.optim.Optimizer`
    """
    # Your code here
    optimizer=torch.optim.RAdam(model.parameters(),
                                lr=0.001,betas=(0.9,0.999),
                                eps=1e-08,weight_decay=0.0001)
    return optimizer


def predict(model, batch):
    """
    model:
        `torch.nn.Module`
        The neural net, as defined by `get_model()`.
    batch:
        unspecified
        A batch of Tiny ImageNet images, as yielded by `get_dataloader(..., 'val')`
        (with same preprocessing and device).

    return:
    prediction:
        `torch.tensor`, shape == (N, 200), dtype == `torch.float32`
        The scores of each input image to belong to each of the dataset classes.
        Namely, `prediction[i, j]` is the score of `i`-th minibatch sample to
        belong to `j`-th class.
        These scores can be 0..1 probabilities, but for better numerical stability
        they can also be raw class scores after the last (usually linear) layer,
        i.e. BEFORE softmax.
    """
    # Your code here
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    batch_device=batch.to(device)
    prediction = model(batch_device)
    return prediction
    
    
    
    
    
def validate(dataloader, model):
    """
    Run `model` through all samples in `dataloader`, compute accuracy and loss.

    dataloader:
        `torch.utils.data.DataLoader` or an object with equivalent interface
        See `get_dataloader()`.
    model:
        `torch.nn.Module`
        See `get_model()`.

    return:
    accuracy:
        `float`
        The fraction of samples from `dataloader` correctly classified by `model`
        (top-1 accuracy). `0.0 <= accuracy <= 1.0`
    loss:
        `float`
        Average loss over all `dataloader` samples.
    """
    # Your code here
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    loss_i = 0.0
    acc_i = 0
    with torch.no_grad():
        for X, Y in tqdm(dataloader):
            X_device=X.to(device)
            Y_device=Y.to(device)
                  
            prediction=model(X_device)
          
            _, preds = torch.max(prediction, 1)


            loss_func=criterion(prediction,Y_device)

            loss_i += loss_func.item() * X_device.size(0)
            acc_i += torch.sum(preds == Y_device.data)

    loss=loss_i / len(dataloader.dataset)
    accuracy = acc_i.double() / len(dataloader.dataset)
    
    return accuracy,loss

    

        


        

def train_on_tinyimagenet(train_dataloader, val_dataloader, model, optimizer):
    """
    Train `model` on `train_dataloader` using `optimizer`. Use best-accuracy settings.

    train_dataloader:
    val_dataloader:
        See `get_dataloader()`.
    model:
        See `get_model()`.
    optimizer:
        See `get_optimizer()`.
    """
    # Your code here
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loss_i = 0.0
    acc_i = 0
    criterion = torch.nn.CrossEntropyLoss()
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    for epoch in range(8):
        model.train()
        for X, Y in tqdm(train_dataloader):
            model.zero_grad()
            X_device=X.to(device)
            Y_device=Y.to(device)
            
            prediction=model(X_device)
          
            _, preds = torch.max(prediction, 1)
            
            loss_func=criterion(prediction,Y_device)
            
            loss_func.backward()
          
            optimizer.step()
            
            loss_i += loss_func.item() * X_device.size(0)
            acc_i += torch.sum(preds == Y_device.data)
        
        epoch_tr_loss=loss_i / len(train_dataloader.dataset)
      
        epoch__tr_acc = acc_i.double() / len(train_dataloader.dataset)
        scheduler.step()
        
        
         
        epoch_val_acc,loss=validate(val_dataloader, model)

        print(f'Training Loss: {epoch_tr_loss:.3f} Acc: {epoch__tr_acc:.3f}')
        print(f'Training Loss: {epoch_tr_loss:.3f} Acc: {epoch__tr_acc:.3f}')


            
      

        

def load_weights(model, checkpoint_path):
    """
    Initialize `model`'s weights from `checkpoint_path` file.

    model:
        `torch.nn.Module`
        See `get_model()`.
    checkpoint_path:
        `str`
        Path to the checkpoint.
    """
    # Your code here
    model.load_state_dict(torch.load(checkpoint_path,map_location='cpu'))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

def get_checkpoint_metadata():
    """
    Return hard-coded metadata for 'checkpoint.pth'.
    Very important for grading.

    return:
    md5_checksum:
        `str`
        MD5 checksum for the submitted 'checkpoint.pth'.
        On Linux (in Colab too), use `$ md5sum checkpoint.pth`.
        On Windows, use `> CertUtil -hashfile checkpoint.pth MD5`.
        On Mac, use `$ brew install md5sha1sum`.
    google_drive_link:
        `str`
        View-only Google Drive link to the submitted 'checkpoint.pth'.
        The file must have the same checksum as in `md5_checksum`.
    """
    # Your code here; 
    md5_checksum = "a1c8b4269f466ddf533086e7b4521ea2"
    # Your code here; 
    google_drive_link = "https://drive.google.com/file/d/1SAMLFWNV5eTeLv6NcMoRuSU8uy0K5Yyj/view?usp=sharing"

    return md5_checksum, google_drive_link
