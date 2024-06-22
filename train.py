import config
import dataloader
import VisionTransformer

import transformers
import torch 
import torch.nn as nn
import numpy as np 
import torchvision

from tqdm import tqdm
import albumentations as alb

def accuracy_fn(target, output):
    output = torch.softmax(output, dim=-1)
    output = output.argmax(dim=-1)
    return ((target==output)*1.0).mean()

def train_fn(model, dataloader, optimizer, scheduler, device):
    running_loss = 0
    running_acc = 0 
    model.train()
    for num_steps, data in tqdm(enumerate(dataloader), total=len(dataloader)):
        for p in model.parameters():
            p.grad = None
        patches = data['patches'].to(device)
        label = data['label'].to(device)
        output = model(patches)
        loss = nn.CrossEntropyLoss()(output, label)
        running_loss += loss.item()
        running_acc += accuracy_fn(label, output).item()
        loss.backward()
        optimizer.step()
        scheduler.step()

    epoch_acc = running_acc/len(dataloader)
    epoch_loss = running_loss/len(dataloader)

    return epoch_acc, epoch_loss

def eval_fn(model, dataloader, device):
    running_loss = 0
    running_acc = 0
    model.eval()
    with torch.no_grad():
        for num_steps, data in tqdm(enumerate(dataloader), total=len(dataloader)):
            patches = data['patches'].to(device)
            label = data['label'].to(device)
            output = model(patches)
            loss = nn.CrossEntropyLoss()(output, label)
            running_loss += loss.item()
            running_acc += accuracy_fn(label, output).item()
    epoch_loss = running_loss/len(dataloader)
    epoch_acc = running_acc/len(dataloader)
    
    return epoch_acc, epoch_loss

def run():
    train_dataset = torchvision.datasets.CIFAR10(root='CIFAR10', train=True, download=True)
    val_dataset = torchvision.datasets.CIFAR10(root='CIFAR10', train=False, download=True)

    train_transform = alb.Compose([
        alb.Resize(config.image_height, config.image_width, always_apply=True),
        alb.Normalize(config.mean, config.std, always_apply=True),
        alb.HorizontalFlip(p=0.1),
        alb.RandomBrightnessContrast(p=0.2),
        #alb.RandomContrast(p=0.1),
        alb.RGBShift(p=0.1),
        alb.GaussNoise(p=0.1),
    ])

    val_transforms = alb.Compose([
        alb.Resize(config.image_height, config.image_width, always_apply=True),
        alb.Normalize(config.mean, config.std, always_apply=True)
    ])

    
    train_data = dataloader.dataloader(train_dataset, train_transform)
    val_data = dataloader.dataloader(val_dataset, val_transforms)


    train_loader = torch.utils.data.DataLoader(
        train_data,
        num_workers=4,
        pin_memory=True,
        batch_size=config.Batch_Size
    )

    val_loader = torch.utils.data.DataLoader(
        val_data,
        num_workers=4,
        pin_memory=True,
        batch_size=config.Batch_Size
    )

    #for num_steps, data in tqdm(enumerate(train_loader), total=len(train_loader)):
    #        patches = data['patches']
    #        print("cifar10 train data shape: "+str(patches.shape))
            
    model = VisionTransformer.ViT(
        patch_height=16,
        patch_width=16,
        embedding_dims = config.embedding_dims,
        dropout = config.dropout,
        num_heads = config.heads,
        num_layers = config.num_layers,
        forward_expansion = config.forward_expansion,
        max_len = config.max_len,
        layer_norm_eps = config.layer_norm_eps,
        num_classes = config.num_classes,
    )
    
    if torch.cuda.is_available():
        accelarator = 'cuda'
    else:
        accelarator = 'cpu'
    
    device = torch.device(accelarator)
    torch.backends.cudnn.benchmark = True

    model = model.to(device)

    optimizer = transformers.AdamW(model.parameters(), lr=config.LR, weight_decay=config.weight_decay)

    num_training_steps = int((config.Epochs*len(train_dataset))/config.Batch_Size)

    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps = int(0.1*num_training_steps),
        num_training_steps = num_training_steps
    )
    
    best_acc = 0
    best_model = 0
    for epoch in range(config.Epochs):
        train_acc, train_loss = train_fn(model, train_loader, optimizer, scheduler, device)
        val_acc, val_loss = eval_fn(model, val_loader, device)
        print(f'\nEPOCH     =  {epoch+1} / {config.Epochs} | LR =  {scheduler.get_last_lr()[0]}')
        print(f'TRAIN ACC = {train_acc*100}% | TRAIN LOSS = {train_loss}')
        print(f'VAL ACC   = {val_acc*100}% | VAL LOSS = {val_loss}')
        if val_acc > best_acc:
            best_acc = val_acc
            best_model = model.state_dict()
            torch.save(best_model, config.Model_Path + 'model_state_dict'+str(epoch)+'.pt') 

if __name__ == "__main__":
    run()