import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler
from torchvision.utils import save_image
from tqdm.auto import tqdm
from model import ClassConditionedUnet
from evaluator import evaluation_model
from dataset import iclevrDataset
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import argparse
import os



class Trainer():
    def __init__(self, args, train_data, test_data):
        # TODO
        self.device = args.device
        self.n_epochs = args.epochs
        self.lr = args.lr
        self.logdir = args.logdir
        self.eval_freq = args.eval_freq
        self.stat_ep = 0

        self.writer = SummaryWriter(self.logdir)
        self.train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=12)
        self.test_dataloader = DataLoader(test_data, batch_size=len(test_data), shuffle=False, num_workers=12)
        
        self.net = ClassConditionedUnet(num_classes=24).to(self.device)
        self.noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule='squaredcos_cap_v2')
        self.loss_fn = nn.MSELoss()
        self.opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        self.evalator = evaluation_model()

        if args.resume != 0:
            self.stat_ep = args.resume
            print("#"*50)
            print(f'resume from {self.logdir}/epoch_{args.resume}.pth ...........')
            print("#"*50)
            self.net.load_state_dict(torch.load(f"{self.logdir}/epoch_{args.resume}.pth"))

    def train(self):
        # TODO
        losses = []
        best_acc = 0
        for epoch in tqdm(range(self.stat_ep, self.n_epochs), f"epoch"):
            for x, y in self.train_dataloader:
                
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
            if epoch % self.eval_freq == 0 and epoch != 0:
                val_acc = self.test(epoch=epoch)
                if val_acc > best_acc:
                    best_acc = val_acc
                    torch.save(self.net.state_dict(), f"{self.logdir}/best.pth")
                torch.save(self.net.state_dict(), f"{self.logdir}/epoch_{epoch}.pth")

    def test(self, epoch=None):
        # TODO
        if not epoch:
            print("#"*50)
            print(f'load from {self.logdir}/best.pth to test.')
            print("#"*50)
            self.net.load_state_dict(torch.load(f"{self.logdir}/best.pth"))

        self.net.eval()
        acc = []
        for _, cond in enumerate(self.test_dataloader):
            cond = cond.to(self.device)
            batch_size = cond.size(0)
            x = torch.randn(batch_size, 3, 64, 64).to(self.device)
            for _, t in enumerate(self.noise_scheduler.timesteps):

                # Get model pred
                with torch.no_grad():
                    residual = self.net(x, t, cond)  # Again, note that we pass in our labels y

                # Update sample with step
                x = self.noise_scheduler.step(residual, t, x).prev_sample

            acc.append(self.evalator.eval(x, cond))
        
        if epoch:
            self.writer.add_scalar("val acc", np.mean(acc), epoch)
            return np.mean(acc)
        else:
            print(f"test accuray: {np.mean(acc)}")
            save_image(x.detach(), f'{self.logdir}/test_{np.mean(acc)*100:.1f}%.png', normalize=True)

def main():
    # TODO
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-d', '--device', default='cuda')
    parser.add_argument('--logdir', default='log/SimpleUnet')
    # train
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--eval_freq', default=5, type=int)
    parser.add_argument('--resume', default=0, type=int)
    # test
    parser.add_argument('--test_only', action='store_true')

    args = parser.parse_args()
    
    os.makedirs(args.logdir, exist_ok=True)
    train_data = iclevrDataset(mode='train', root='dataset')
    test_data = iclevrDataset(mode='test', root='dataset')
    trainer = Trainer(args, train_data, test_data)

    if not args.test_only:
        trainer.train()

    trainer.test()

if __name__ == '__main__':
    main()
