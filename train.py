from tqdm import tqdm
import torch
from model import UNet
from torch import nn
from utils import *




device='cuda' if torch.cuda.is_available() else 'cpu'
train_path=r'D:\data\CRACK500\traindata\traindata'
test_path=r'D:\data\CRACK500\testdata\testdata'
pin_memory = True
lr = 1e-4
num_epochs = 3
batch_size=16
num_of_workers=4
epochs=50
upper_loss=1.0
import warnings
warnings.filterwarnings("ignore", category=UserWarning) 



model=UNet().to(device)

loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=.0001)
scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=0.1,patience=2,verbose=True)

checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer":optimizer.state_dict(),
            'scheduler':scheduler.state_dict()
        }


train_loader,test_loader=get_loader(train_path,test_path,batch_size,num_of_workers,pin_memory)

def train(train_loader,model,loss_fn,optimizer,device):
  model.train()
  loop=tqdm(train_loader)
  total_acc=0
  total_loss=0
  total_dice=0
  for id,(x,y) in enumerate(loop):
    x,y=x.to(device),y.to(device)
    y_logits=model(x)
    loss=loss_fn(y_logits,y)
    preds=(y_logits.sigmoid()>0.5).float()
    accuracy=(preds==y).sum()/torch.numel(preds)
    dice= (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)
    total_dice+=dice

    total_loss+=loss
    total_acc+=accuracy
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    loop.set_postfix(loss=loss.item(),accuracy=accuracy.item(),dice_score=dice.item())
    loop.set_description(f'train')

  avg_loss=total_loss/len(train_loader)
  avg_acc=total_acc/len(train_loader)
  avg_dice=total_dice/len(test_loader)

  return avg_loss,avg_acc,avg_dice

def test(test_loader,model,loss_fn,device):
  model.eval()
  loop=tqdm(test_loader)
  total_acc=0
  total_loss=0
  total_dice=0
  with torch.inference_mode():
    for num,(x,y) in enumerate(loop):
      x,y=x.to(device),y.to(device)
      y_logits=model(x)
      loss=loss_fn(y_logits,y)
      preds=(y_logits.sigmoid()>0.5).float()
      accuracy=(preds==y).sum()/torch.numel(preds)
      total_loss+=loss
      total_acc+=accuracy
      dice= (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)
      total_dice+=dice
      id=num+1
      loop.set_postfix(loss=loss.item()/id,accuracy=accuracy.item(),dice_score=dice.item())
      loop.set_description(f'test')


    avg_loss=total_loss/len(test_loader)
    avg_acc=total_acc/len(test_loader)
    avg_dice=total_dice/len(test_loader)

    return avg_loss,avg_acc,avg_dice

def main():
    result_train={'loss':[],'accuracy':[],'dice':[]}
    result_test={'loss':[],'accuracy':[],'dice':[]}
    model.load_state_dict(torch.load(r"D:\latest_model_collab.pth")['state_dict'])
    optimizer.load_state_dict(torch.load(r"D:\latest_model_collab.pth")['optimizer'])
    scheduler.load_state_dict(torch.load(r"D:\latest_model_collab.pth")['scheduler'])
    for epoch in range(epochs):
        avg_loss,avg_acc,avg_dice=train(train_loader,model,loss_fn,optimizer,device)
        scheduler.step(avg_loss)
        avg_test_loss,avg_test_acc,avg_test_dice=test(test_loader,model,loss_fn,device)
        print(f'{epoch}/{epochs}'+'--'*50)
        if (epoch+1) %10==0:
            save_latest(checkpoint)
            print('latest model has been saved')
        # if avg_loss<upper_loss:
        #     upper_loss=avg_loss
        #     save_best(upper_loss)
            # print(f'best model has been saved with loss of {upper_loss}')
    result_train['loss'].append(avg_loss),result_train['accuracy'].append(avg_acc),result_train['dice'].append(avg_dice)
    result_test['loss'].append(avg_test_loss),result_test['accuracy'].append(avg_test_acc),result_test['dice'].append(avg_test_dice)
if __name__ == "__main__":
    main()