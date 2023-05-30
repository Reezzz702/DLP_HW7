from dataclasses import dataclass
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler
from dataset import iclevrDataset
from diffusers.optimization import get_cosine_schedule_with_warmup
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from torchvision.utils import make_grid, save_image
from evaluator import evaluation_model
import torch.nn as nn
from model import *
import argparse
from torch.utils.tensorboard import SummaryWriter

def train(args, model, noise_scheduler, optimizer, train_dataloader, test_dataloader, lr_scheduler=None, writer=None):
	global_step = 0
	highest_acc = 0.0
	
	for epoch in tqdm(range(args.resume, args.num_epochs)):
		total_loss = 0.0
		for step, batch in enumerate(train_dataloader):
			clean_images, labels = batch
			clean_images = clean_images.to(args.device)
			labels = labels.to(args.device)

			# Sample noise to add to the images
			bs = clean_images.shape[0]
			noise = torch.randn_like(clean_images)
			timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bs,)).to(clean_images.device).long()
			# Add noise to the clean images according to the noise magnitude at each timestep
			# (this is the forward diffusion process)
			noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

			pred = model(noisy_images, timesteps, labels)

			loss = F.mse_loss(pred, noise)
			loss.backward()
			optimizer.step()
			if lr_scheduler is not None:
				lr_scheduler.step()
			optimizer.zero_grad()
			
			global_step += 1
			total_loss += loss.detach().item()

		writer.add_scalar("train loss", total_loss/len(train_dataloader), epoch)
		if (epoch + 1) % args.eval_freq == 0 or epoch == args.num_epochs - 1:
			acc = val(args, test_dataloader, model, noise_scheduler)
			writer.add_scalar("val acc", acc, epoch)
			if highest_acc < acc:
				print("Highest accuracy, Saving best model!!!")
				highest_acc = acc
				torch.save(model.state_dict(), f"{args.logdir}/{args.unet}Unet/best.pth")

			torch.save(model.state_dict(), f"{args.logdir}/{args.unet}Unet/epoch_{epoch+1}.pth")

def val(args, test_dataloader, model, noise_scheduler):
	evaluator = evaluation_model()
	if args.test_only:
		print("#"*50)
		print(f'load from {args.logdir}/{args.unet}Unet/best.pth to test.')
		print("#"*50)
		model.load_state_dict(torch.load(f"{args.logdir}/{args.unet}Unet/best.pth"))

	model.eval()
	total_acc = 0.0
	for i, cond in enumerate(test_dataloader):
		cond = cond.to(args.device)
		bs = cond.shape[0]
		x = torch.randn(bs, 3, 64, 64).to(args.device)
		for _, t in enumerate(noise_scheduler.timesteps):
			with torch.no_grad():
				residual = model(x, t, cond)
			x = noise_scheduler.step(residual, t, x).prev_sample
		acc = evaluator.eval(x, cond)
		total_acc += acc
	print("Average accuracy: ", total_acc/len(test_dataloader))
	
	return total_acc/len(test_dataloader)

def test(args, test_dataloader, model, noise_scheduler):
	evaluator = evaluation_model()
	if args.test_only:
		print("#"*50)
		print(f'load from {args.logdir}/{args.unet}Unet/best.pth to test.')
		print("#"*50)
		model.load_state_dict(torch.load(f"{args.logdir}/{args.unet}Unet/best.pth"))

	best_acc = 0
	acc_list = []
	for _ in tqdm(range(20)):
		model.eval()
		total_acc = 0.0
		for i, cond in enumerate(test_dataloader):
			cond = cond.to(args.device)
			bs = cond.shape[0]
			x = torch.randn(bs, 3, 64, 64).to(args.device)
			for _, t in enumerate(noise_scheduler.timesteps):
				with torch.no_grad():
					residual = model(x, t, cond)
				x = noise_scheduler.step(residual, t, x).prev_sample
			acc = evaluator.eval(x, cond)
			total_acc += acc

		temp_acc = total_acc/len(test_dataloader)
		if temp_acc > best_acc:
			best_acc = temp_acc
			save_image(x.detach(), f'{args.logdir}/{args.unet}Unet/{args.test_file}_{np.mean(acc)*100:.1f}%.png', normalize=True)
		
		acc_list.append(temp_acc)

	print("Average accuracy: ", np.mean(acc_list))
	print("Best acc: ", best_acc)

	

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('-d', '--device', default='cuda')
	parser.add_argument('--logdir', default='log')
	# train
	parser.add_argument('--unet', default="base", type=str)
	parser.add_argument('--num_epochs', default=150, type=int)
	parser.add_argument('--batch_size', default=32, type=int)
	parser.add_argument('--lr', default=1e-4, type=float)
	parser.add_argument('--eval_freq', default=25, type=int)
	parser.add_argument('--resume', default=0, type=int)
	parser.add_argument('--lr_warmup_steps', default=500, type=float)
	parser.add_argument('--scheduler', default='linear', type=str)
	# test
	parser.add_argument('--test_only', action='store_true')
	parser.add_argument('--test_file', default='test', type=str)

	args = parser.parse_args()

	if args.scheduler != 'linear':
		args.logdir = f"{args.logdir}/{args.scheduler}"

	train_dataset = iclevrDataset(mode='train', root='./dataset/')
	test_dataset = iclevrDataset(mode='test', root='./dataset/', test_file=f"{args.test_file}.json")

	train_dataloader = DataLoader(
		train_dataset, 
		batch_size=args.batch_size,
		shuffle=True,
		num_workers=12)
	test_dataloader = DataLoader(
		test_dataset,
		batch_size=32,
		shuffle=False,
		num_workers=12)
	
	if args.unet == "base":
		model = BaseConditionedUnet().to(args.device)
	elif args.unet == "shallow":
		model = ShallowConditionedUnet().to(args.device)

	if args.resume != 0:
			stat_ep = args.resume
			print("#"*50)
			print(f'resume from {args.logdir}/{args.unet}Unet/epoch_{args.resume}.pth ...........')
			print("#"*50)
			model.load_state_dict(torch.load(f"{args.logdir}/{args.unet}Unet/epoch_{args.resume}.pth"))

	# setup diffuser
	noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule=args.scheduler)
	optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
	lr_scheduler = get_cosine_schedule_with_warmup(
		optimizer=optimizer,
		num_warmup_steps=args.lr_warmup_steps,
		num_training_steps=(len(train_dataloader) * args.num_epochs),
	)
	writer = SummaryWriter(f"{args.logdir}/{args.unet}Unet")

	if not args.test_only:
		train(args, model, noise_scheduler, optimizer, train_dataloader, test_dataloader, lr_scheduler, writer)
	else:
		test(args, test_dataloader, model, noise_scheduler)