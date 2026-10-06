from torchvision import transforms

def get_base_transforms(image_size: int = 224) -> transforms.Compose:
    """
    Returns the foundational transformations required for all splits:
    resizing, conversion to tensor, and normalization.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        # Standard ImageNet normalization (suitable since we'll use pre-trained backbones)
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
