import albumentations as A
from albumentations.pytorch import ToTensorV2

def gst_train_transforms(image_size=(224, 224)):
    return A.Compose([
        A.Resize(height=image_size[0], width=image_size[1]),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p-0.5),
        A.RandomRotate90(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.5),
        A.Normalize(MEAN=(0.485, 0.456, 0.409), STD=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

def get_val_transforms(image_size=(224, 224)):
    return A.Compose([
        A.Resize(height=image_size[0], width=image_size[1]),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2() 
    ])