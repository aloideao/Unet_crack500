import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import glob
from augment import albumentations
import cv2
class crackdataset(Dataset):
    def __init__(self, image_dir,augment=True):
        self.image_dir = image_dir
        self.transform=albumentations(augment)
 

        self.images = glob.glob(os.path.join(self.image_dir,'*.jpg'))
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = self.images[index]
        mask_path = img_path.replace(".jpg", '_mask.png')


        image =cv2.imread(img_path)[:,:,::-1].astype(np.uint8)
        mask = cv2.imread(mask_path,0).astype(np.uint8)

        if image.shape[0]!=mask.shape[0]:
             image=np.rot90(image,k=1)
        mask=np.expand_dims(mask,-1)
        mask[mask == 255.0] = 1.0

        if self.transform is not None:
            augmentations = self.transform(image,mask)
            image = augmentations["image"]
            mask = augmentations["mask"].permute(2,0,1).float()

        return image, mask