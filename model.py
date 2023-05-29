import torch
from torch import nn
from diffusers import UNet2DModel


class ShallowConditionedUnet(nn.Module):
	def __init__(self):
		super().__init__()
	
		# Self.model is an unconditional UNet with extra input channels to accept the conditioning information (the class embedding)
		self.model = UNet2DModel(
			sample_size=64,           # the target image resolution
			in_channels=3 + 24, # Additional input channels for class cond.
			out_channels=3,           # the number of output channels
			layers_per_block=2,       # how many ResNet layers to use per UNet block
			block_out_channels=(128, 128, 256, 256), 
			down_block_types=( 
				"DownBlock2D",        # a regular ResNet downsampling block
				"DownBlock2D",        # a regular ResNet downsampling block				
				"AttnDownBlock2D",    # a ResNet downsampling block with spatial self-attention
				"DownBlock2D",        # a regular ResNet downsampling block				
				# "AttnDownBlock2D",
			), 
			up_block_types=(
				# "AttnUpBlock2D", 
				"UpBlock2D",          # a regular ResNet upsampling block
				"AttnUpBlock2D",      # a ResNet upsampling block with spatial self-attention
				"UpBlock2D",          # a regular ResNet upsampling block
				"UpBlock2D",          # a regular ResNet upsampling block
			),
		)

	# Our forward method now takes the class labels as an additional argument
	def forward(self, x, t, class_cond):
		# Shape of x:
		bs, ch, w, h = x.shape
		
		class_cond = class_cond.view(bs, class_cond.shape[1], 1, 1).expand(bs, class_cond.shape[1], w, h)
		# x is shape (bs, 1, 28, 28) and class_cond is now (bs, 4, 28, 28)

		# Net input is now x and class cond concatenated together along dimension 1
		net_input = torch.cat((x, class_cond), 1) # (bs, 5, 28, 28)

		# Feed this to the unet alongside the timestep and return the prediction
		return self.model(net_input, t).sample # (bs, 1, 28, 28)
	
class BaseConditionedUnet(nn.Module):
	def __init__(self, num_classes=24):
		super().__init__()

		# self.model is an unconditional Unet with extra channels for class conditioning
		self.model = UNet2DModel(
			sample_size=64,  # the target image resolution
			in_channels=3+num_classes,  # the number of input channels, 3 for RGB images
			out_channels=3,  # the number of output channels
			layers_per_block=2,  # how many ResNet layers to use per UNet block
			block_out_channels=(128, 128, 256, 256, 512, 512),  # the number of output channels for each UNet block
			down_block_types=(
				"DownBlock2D",  # a regular ResNet downsampling block
				"DownBlock2D",
				"DownBlock2D",
				"DownBlock2D",  # a ResNet downsampling block with spatial self-attention
				"AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
				"DownBlock2D",
			),
			up_block_types=(
				"UpBlock2D",  # a regular ResNet upsampling block
				"AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
				"UpBlock2D",  # a ResNet upsampling block with spatial self-attention
				"UpBlock2D",
				"UpBlock2D",
				"UpBlock2D",
			),
		)
	
	def forward(self, x, t, cond):
		# Shape os x:
		bs, ch, w, h = x.shape
		cond = cond.view(bs, cond.shape[1], 1, 1).expand(bs, cond.shape[1], w, h)

		net_input = torch.cat([x, cond], dim=1)
		output = self.model(sample=net_input, timestep=t).sample

		return output