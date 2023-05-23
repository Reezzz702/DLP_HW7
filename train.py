import torch
import torchvision
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler, UNet2DModel
from matplotlib import pyplot as plt
from tqdm.auto import tqdm
from model import ClassConditionedUnet
from torch.utils.tensorboard import SummaryWriter



class Trainer():
    def __init__(self, args, train_data, test_data, writer):
        # TODO
        self.device = args.device
        self.n_epochs = args.epochs
        self.lr = args.lr

        self.writer = writer
        self.train_dataloader = DataLoader(train_data, batch_size=128, shuffle=True)
        self.test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)
        
        self.net = ClassConditionedUnet().to(self.device)
        self.noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule='squaredcos_cap_v2')
        self.loss_fn = nn.MSELoss()
        self.opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)

    def train(self):
        # TODO
        losses = []
        for epoch in range(self.n_epochs):
            for x, y in tqdm(self.train_dataloader):
                
                # Get some data and prepare the corrupted version
                x = x.to(self.device) * 2 - 1 # Data on the GPU (mapped to (-1, 1))
                y = y.to(self.device)
                noise = torch.randn_like(x)
                timesteps = torch.randint(0, 999, (x.shape[0],)).long().to(self.device)
                noisy_x = self.noise_scheduler.add_noise(x, noise, timesteps)

                # Get the model prediction
                pred = self.net(noisy_x, timesteps, y) # Note that we pass in the labels y

                # Calculate the loss
                loss = self.loss_fn(pred, noise) # How close is the output to the noise

                # Backprop and update the params:
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                # Store the loss for later
                losses.append(loss.item())

            # Print our the average of the last 100 loss values to get an idea of progress:
            avg_loss = sum(losses[-100:])/100
            self.writer.add_scalar("train loss", avg_loss, epoch)
            # print(f'Finished epoch {epoch}. Average of the last 100 loss values: {avg_loss:05f}')

    def test(self):
        # TODO
        pass
