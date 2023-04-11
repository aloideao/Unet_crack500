from albumentations.pytorch.transforms import ToTensorV2
import albumentations as A
class albumentations:
    def __init__(self,augment=True):
      T_train=[
                  A.Resize(height=224, width=224),
                  A.Rotate(limit=35, p=1.0),
                  A.HorizontalFlip(p=0.5),
                  A.VerticalFlip(p=0.1),
                  A.Normalize(
                      mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225],
                      max_pixel_value=255.0,
                  ),
                  ToTensorV2(),
              ]

      T_val=[
                  A.Resize(height=224, width=224),
                  A.Normalize(
                      mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225],
                      max_pixel_value=255.0,
                  ),
                  ToTensorV2(),
              ]
          
      T=T_train if augment else T_val 
      self.transfrom=A.Compose(T)


    def __call__(self,img,mask):

        return self.transfrom(image=img,mask=mask)
