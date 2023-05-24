from dataloader import *
from model import *

input_folder = './DRIVE/training/images'
label_folder = './DRIVE/training/1st_manual'
train_data_loader = get_DRIVE_Dataloader(input_folder=input_folder, 
                                  label_folder=label_folder, 
                                  batch_size=4)

# device = torch.device("mps")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = UNet(in_channels=3, out_channels=1)
model.to(device,dtype=torch.float32)

optimizer = get_optimiazer(model)
loss_fn = dice_loss
num_epochs = 200
save_path = './checkpoint/DRIVE.pth'

trainer(train_loader=train_data_loader, 
        model=model, optimizer=optimizer, loss_fn=loss_fn, 
        num_epochs=num_epochs,
        device=device, save_path=save_path)
