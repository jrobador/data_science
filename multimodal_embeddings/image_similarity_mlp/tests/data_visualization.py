import torch
from torchvision import datasets, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

#Images transformation
image_transform = transforms.Compose([
    #transforms.RandomResizedCrop(32, interpolation=Image.BICUBIC),
    #transforms.RandomHorizontalFlip(p=0.5),
    #transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    #transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
])

#Custom Dataloader
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

train_dataset = CustomCIFAR10(root='./data', train=True, download=False, transform=image_transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)

print (train_dataset[0][1])

num_images_to_display = 10

for i in range(num_images_to_display):

    image, label = train_dataset[30+i]
    image_np = image.numpy().transpose((1, 2, 0))
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
    image_np = std * image_np + mean
    image_np = np.clip(image_np, 0, 1)
    plt.subplot(1, num_images_to_display, i + 1)
    plt.imshow(image_np)
    plt.title(f"Label: {label}")
    plt.axis('off')

plt.show()