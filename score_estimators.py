import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

class LEScore(nn.Module):
	def __init__(self, 
				dataset,
				schedule,
				kernel_size : int = 3,
				batch_size :int = 32,
				max_samples : int = 10000,
				conditional : bool = False):
		"""
		Learns the ideal score for each pixel given image patches.
		"""
		super().__init__()

		self.dataset = dataset
		self.schedule = schedule
		self.kernel_size = kernel_size
		self.batch_size = batch_size
		self.max_samples = max_samples
		self.conditional = conditional
		
		self.trainloader = DataLoader(self.dataset, batch_size=batch_size)
		self.pad = self.kernel_size // 2
		
		# Precompute the patches, as it does not depend on the labels
		if not self.conditional:
			device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
			seen = 0
			patches_list, norms_list, centers_list = [], [], []

			for images, labels in self.trainloader:
				# Break the loop if the max samples are reached
				seen += images.shape[0]
				if self.max_samples is not None and seen > self.max_samples:
					break

				images = images.to(device)
				
				# Pad the images 
				images = F.pad(images, (self.pad, self.pad, self.pad, self.pad), mode='constant', value=0)
	
				# Get the patches from this batch of images
				patches = images.unfold(2, self.kernel_size, 1).unfold(3, self.kernel_size, 1) # [B, c, h', w', k, k]
				rolled = patches.permute(0, 2, 3, 1, 4, 5) # [B, h', w', c, k, k]
				# Reshape the patches to be a collection of all the patches from the images
				patches = rolled.reshape(rolled.shape[0]*rolled.shape[1]*rolled.shape[2], 
										rolled.shape[3], 
										self.kernel_size, 
										self.kernel_size) # [NP, c, k, k], NP = number of patches
				
				# Squared L2 Norm squares of the patches
				pnorms = torch.sum(patches**2, dim=(1,2,3)) # [NP]
				# Center of the patches
				pcenters = patches[:,:,self.pad,self.pad] # [NP, c] 

				patches_list.append(patches)
				norms_list.append(pnorms)
				centers_list.append(pcenters)
			
			patches = torch.cat(patches_list, dim=0) # [NP, C, k, k]
			pnorms = torch.cat(norms_list, dim=0) # [NP]
			pcenters = torch.cat(centers_list, dim=0) # [NP, C]
			self.register_buffer('patches',   patches)
			self.register_buffer('pnorms',    pnorms)
			self.register_buffer('pcenters',  pcenters)

	def forward(self, x, t, label=None):
		"""
		Computes the ideal score for each pixel in the input at time t

		Args:	
			x: input image tensor [B, C, H, W]
			t: time step 
			label: optional label to filter training samples

		Returns:
			score tensor [B, C, H, W]
		"""
		device = x.device
		b, c, h, w = x.shape

		bt = self.schedule(t).sqrt().to(device)
		at = (1-self.schedule(t)).sqrt().to(device)

		# Pad the input image with 0s
		xpadded = F.pad(x, (self.pad, self.pad, self.pad, self.pad), mode='constant', value=0) 
		# Get the patches from the padded image
		xpatches = xpadded.unfold(2, self.kernel_size, 1).unfold(3, self.kernel_size, 1) # [B, c, h', w', k, k], h'=h+2d, w'=w+2d
		# Squared L2 Norm squared of the patches for each pixel 
		xnorms = xpatches.pow(2).sum(dim=(1, 4, 5)) # [B, h, w]

		# The numerator and denominator for the score
		numerator = torch.zeros(x.shape, device=device) # [B, c, h, w]
		denominator = torch.zeros(b,h,w, device=device) # [B, h, w]
		subtraction = None

		if self.conditional:
			seen = 0 # Number of data from the dataset seen so far
			for i, (images, labels) in enumerate(self.trainloader):
				# Break the loop if the max samples are reached
				seen += images.shape[0]
				if self.max_samples is not None and seen > self.max_samples:
					break

				# Filtering the images of the dataset based on the label
				if label is not None:
					mask = (labels == label).squeeze()
					images = images[mask]
				if images.shape[0] == 0:
					continue

				images = images.to(device)
				# Pad the images 
				images = F.pad(images, (self.pad, self.pad, self.pad, self.pad), mode='constant', value=0)
	
				# Get the patches from this batch of images
				patches = images.unfold(2, self.kernel_size, 1).unfold(3, self.kernel_size, 1) # [B, c, h', w', k, k]
				rolled = patches.permute(0, 2, 3, 1, 4, 5) # [B, h', w', c, k, k]
				# Reshape the patches to be a collection of all the patches from the images
				patches = rolled.reshape(rolled.shape[0]*rolled.shape[1]*rolled.shape[2], 
										rolled.shape[3], 
										self.kernel_size, 
										self.kernel_size) # [NP, c, k, k], NP = number of patches
				
				# Squared L2 Norm squares of the patches
				pnorms = torch.sum(patches**2, dim=(1,2,3)) # [NP]
				# Center of the patches
				pcenters = patches[:,:,self.pad,self.pad] # [NP, c] 
					
				# The dot product of the patches and the image
				pdotx = F.conv2d(xpadded, patches, padding=0) # [B, NP, H, W]
				
				# Expansion of the dot product to get the exp part of the gaussian
				exp_args = -(xnorms[:,None,:,:] - 2*at*pdotx + (at**2)*pnorms[None,:,None,None])/(2*bt.pow(2)) # [B, NP, h,w]  

				# To keep the computation stable
				if subtraction is None:
					subtraction = torch.amax(exp_args, dim=1, keepdim=True)
				else:
					new_subtraction = torch.amax(exp_args, dim=1, keepdim=True)
					delta_subtraction = (new_subtraction>subtraction)*new_subtraction+(subtraction>=new_subtraction)*subtraction # Mask
					numerator /= torch.exp(delta_subtraction-subtraction) # [B, c, h, w]
					denominator /= torch.exp(delta_subtraction-subtraction)[:,0,:,:] # [B, h, w]
					subtraction = delta_subtraction

				# The exponential part of the gaussian
				exp_vals = torch.exp(exp_args - subtraction) # [B, NP, h, w] 
				# The difference of the center of the noised patches and the image
				num_vals = (x[:,None,:,:,:] - at*pcenters[None,:,:,None,None]) # [B, NP, c, h, w]
				
				numerator += torch.sum(exp_vals[:,:,None,:,:]*num_vals, dim=1)
				denominator += torch.sum(exp_vals, dim=1)
		else:
			NP = self.patches.shape[0]
			for start in range(0, NP, self.patch_batch_size):
				end = start + self.patch_batch_size
				patches = self.patches[start:end]
				pnorms = self.pnorms[start:end]
				pcenters = self.pcenters[start:end]
					
				# The dot product of the patches and the image
				pdotx = F.conv2d(xpadded, patches, padding=0) # [B, NP, H, W]
				
				# Expansion of the dot product to get the exp part of the gaussian
				exp_args = -(xnorms[:,None,:,:] - 2*at*pdotx + (at**2)*pnorms[None,:,None,None])/(2*bt.pow(2)) # [B, NP, h,w]  

				# To keep the computation stable
				if subtraction is None:
					subtraction = torch.amax(exp_args, dim=1, keepdim=True)
				else:
					new_subtraction = torch.amax(exp_args, dim=1, keepdim=True)
					delta_subtraction = (new_subtraction>subtraction)*new_subtraction+(subtraction>=new_subtraction)*subtraction # Mask
					numerator /= torch.exp(delta_subtraction-subtraction) # [B, c, h, w]
					denominator /= torch.exp(delta_subtraction-subtraction)[:,0,:,:] # [B, h, w]
					subtraction = delta_subtraction

				# The exponential part of the gaussian
				exp_vals = torch.exp(exp_args - subtraction) # [B, NP, h, w] 
				# The difference of the center of the noised patches and the image
				num_vals = (x[:,None,:,:,:] - at*pcenters[None,:,:,None,None]) # [B, NP, c, h, w]
				
				numerator += torch.sum(exp_vals[:,:,None,:,:]*num_vals, dim=1)
				denominator += torch.sum(exp_vals, dim=1)
				
		return -numerator/(denominator[:,None,:,:]* bt.pow(2)) # [B, c, h, w]


class IdealScore(nn.Module):
	def __init__(self, 
				dataset, 
				schedule,
				max_samples :int = 10000, 
				):
		"""
		Computes the exact score for each pixel of the input
		"""
		super().__init__()
  
		self.dataset = dataset
		self.max_samples = max_samples
		self.schedule = schedule

		self.trainloader = DataLoader(self.dataset, batch_size=self.max_samples)

	def forward(self, x, t, label=None): 
		"""
		Computes the ideal score for the given input x at time t

		Args:
			x: noised input images [B, C, H, W]
			t: time-step
			label: optional label to filter training samples

		Returns:
			score tensor [B, C, H, W]
		"""
		device = x.device
		x = x.to(device)
 
		bt = self.schedule(t).sqrt().to(device)
		at = (1-self.schedule(t)).sqrt().to(device)

		images, labels = next(iter(self.trainloader)) # [B, c, h, w], B = max_samples

		# Filtering the images of the dataset based on the label
		if label is not None:
			mask = (labels == label).squeeze()
			images = images[mask]

		images = images.to(device)

		# The part in the numerator for the difference of the noised images and the input image
		pwise_diffs = x[:,None,:,:,:]-at*images[None,:,:,:,:] # [B, NP, c, h,w] 
		# The exponential part of the gaussian 
		weight_args = -torch.sum(pwise_diffs**2, dim=(2,3,4))/(2*bt**2) # [B, NP] 
		# The weights for each of the patches
		weights = torch.softmax(weight_args,dim=1) # [B, NP] 

		# The weighted sum of the differences, sigma(diff * weights)
		qkv = torch.sum(pwise_diffs*weights[:,:,None,None,None], dim=1)

		# Negative because images-x, bt**2 = 1 - alpha_bar_t
		return -qkv/bt.pow(2) # [B, c, h, w]  