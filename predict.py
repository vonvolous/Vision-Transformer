import config
import VisionTransformer
import train

import torch
import torch.nn as nn
import numpy as np
from PIL import Image 
import albumentations as alb
import torchvision
import dataloader


def predict(image_path):
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

    model.load_state_dict(torch.load('./model2/model_state_dict196.pt'))
    model.eval()


    test_dataset = torchvision.datasets.CIFAR10(root='CIFAR10', train=False, download=True)

    test_transforms = alb.Compose([
        alb.Resize(config.image_height, config.image_width, always_apply=True),
        alb.Normalize(config.mean, config.std, always_apply=True)
    ])

    test_data = dataloader.dataloader(test_dataset, test_transforms)

    test_loader = torch.utils.data.DataLoader(
        test_data,
        num_workers=4,
        pin_memory=True,
        batch_size=config.Batch_Size
    )
    
    if torch.cuda.is_available():
        accelarator = 'cuda'
    else:
        accelarator = 'cpu'
    
    device = torch.device(accelarator)
    torch.backends.cudnn.benchmark = True

    model = model.to(device)
    
    test_acc, test_loss = train.eval_fn(model, test_loader, device)
    print(f'TEST ACC   = {test_acc*100}% | TEST LOSS = {test_loss}')

    image = np.array(Image.open(image_path).convert('RGB'))
    transform = alb.Compose([
        alb.Resize(config.image_height, config.image_width, always_apply=True),
        alb.Normalize(config.mean, config.std, always_apply=True)
    ])

    image = transform(image=image)['image']
    
    image = torch.tensor(image, dtype=torch.float)
    #image = image.unfold(0, config.patch_size, config.patch_size).unfold(1, config.patch_size, config.patch_size)
    #image = image.reshape(image.shape[0], image.shape[1], image.shape[2]*image.shape[3]*image.shape[4])
    #patches = image.view(-1, image.shape[-1])
    image = image.permute(2, 0, 1)  # 이미지를 [3, 224, 224] 형태로 변환합니다.
    patches = image.unsqueeze(0)

    idx_to_class = {
        0: 'airplane',
        1: 'automobile',
        2: 'bird',
        3: 'cat',
        4: 'deer',
        5: 'dog',
        6: 'frog',
        7: 'horse',
        8: 'ship',
        9: 'truck'
    }
    with torch.no_grad():
        output = model(patches)
    
    prediction_class = torch.softmax(output, dim=-1)[0].argmax(dim=-1).item()
    prediction = idx_to_class[prediction_class]
    print(f'THE IMAGE CONTAINS A {prediction.upper()}')

if __name__ == "__main__":
    image_path = './testImage/image.jpg'
    predict(image_path)