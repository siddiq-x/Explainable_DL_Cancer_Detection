from torchvision import transforms

def get_train_transforms(image_size: int = 224) -> transforms.Compose:
    """
    Returns training-specific transformations, including label-preserving
    data augmentations suited for medical images.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        # Mild augmentations that do not alter the pathology context significantly
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(contrast=0.2, brightness=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
