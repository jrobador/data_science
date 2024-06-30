import torch
from torchvision import datasets, transforms
from PIL import Image

#Images transformation
image_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
])

#Custom CIFAR10
class CustomCIFAR10(datasets.CIFAR10):
    def __init__(self, *args, **kwargs):
        super(CustomCIFAR10, self).__init__(*args, **kwargs)

    def __getitem__(self, index):
        image, label = self.data[index], self.targets[index]
        image = Image.fromarray(image)
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return image, label

train_dataset = CustomCIFAR10(root='./data', train=True, download=True, transform=image_transform)
train_loader  = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
val_dataset   = CustomCIFAR10(root='./data', train=False, download=True, transform=image_transform)
val_loader    = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)