from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.transforms.functional as F


def load_image(image_path):


    return Image.open(image_path).convert('RGB')


def apply_transforms(image, size=224):
    

    if not isinstance(image, Image.Image):
        image = F.to_pil_image(image)

    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(means, stds)
    ])

    tensor = transform(image).unsqueeze(0)

    # tensor.requires_grad = True

    return tensor


def apply_crop_resize(image, size=224):
    

    if not isinstance(image, Image.Image):
        image = F.to_pil_image(image)

    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
    ])

    return transform(image)

