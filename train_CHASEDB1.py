from dataloader import *
from model import *

folder = './CHASEDB1/'
train_data_loader = get_CHASEDB1_Dataloader(folder=folder, batch_size=4)

# device = torch.device("mps")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = UNet(in_channels=3, out_channels=1)
model.to(device,dtype=torch.float32)

optimizer = get_optimiazer(model)
loss_fn = dice_loss
num_epochs = 200
save_path = './checkpoint/CHASEDB1.pth'

trainer(train_loader=train_data_loader, 
        model=model, optimizer=optimizer, loss_fn=loss_fn, 
        num_epochs=num_epochs,
        device=device, save_path=save_path)
