import torch
from torch.nn import Module
import torchvision.transforms
import time


RandomHorizontalFlip = torchvision.transforms.RandomHorizontalFlip
RandomVerticalFlip   = torchvision.transforms.RandomVerticalFlip
RandomCrop           = torchvision.transforms.RandomCrop
AutoAugment          = torchvision.transforms.AutoAugment
Resize               = torchvision.transforms.Resize
GaussianBlur         = torchvision.transforms.GaussianBlur
ColorJitter          = torchvision.transforms.ColorJitter
RandomInvert         = torchvision.transforms.RandomInvert
Pad                  = torchvision.transforms.Pad

def RandomAffine(**kwargs):
    if 'interpolation' in kwargs:
        kwargs['interpolation'] = torchvision.transforms.InterpolationMode(kwargs['interpolation'])
    return torchvision.transforms.RandomAffine(**kwargs)


class AddGaussianNoise(Module):
    """
    Adds Gaussian noise to the input tensor.
    RH 2021
    """
    def __init__(self, mean=0., std=1., level_bounds=(0., 1.), prob=1):
        """
        Initializes the class.
        Args:
            mean (float): 
                The mean of the Gaussian noise.
            std (float):
                The standard deviation of the Gaussian 
                 noise.
            level_bounds (tuple):
                The lower and upper bound of how much
                 noise to add.
            prob (float):
                The probability of adding noise at all.
        """
        super().__init__()

        self.std = std
        self.mean = mean

        self.prob = prob
        
        self.level_bounds = level_bounds
        self.level_range = level_bounds[1] - level_bounds[0]

    def forward(self, tensor):
        if torch.rand(1) <= self.prob:
            level = torch.rand(1, device=tensor.device) * self.level_range + self.level_bounds[0]
            return (1-level)*tensor + level*(tensor + torch.randn(tensor.shape, device=tensor.device) * self.std + self.mean)
        else:
            return tensor
    def __repr__(self):
        return f"AddGaussianNoise(mean={self.mean}, std={self.std}, level_bounds={self.level_bounds}, prob={self.prob})"

class AddPoissonNoise(Module):
    """
    Adds Poisson noise to the input tensor.
    RH 2021
    """
    def __init__(self, scaler_bounds=(0.1,1.), prob=1, base=10, scaling='log'):
        """
        Initializes the class.
        Args:
            lam (float): 
                The lambda parameter of the Poisson noise.
            scaler_bounds (tuple):
                The bounds of how much to multiply the image by
                 prior to adding the Poisson noise.
            prob (float):
                The probability of adding noise at all.
            base (float):
                The base of the logarithm used if scaling
                 is set to 'log'. Larger base means more
                 noise (higher probability of scaler being
                 close to scaler_bounds[0]).
            scaling (str):
                'linear' or 'log'
        """
        super().__init__()

        self.prob = prob
        self.bounds = scaler_bounds
        self.range = scaler_bounds[1] - scaler_bounds[0]
        self.base = base
        self.scaling = scaling

    def forward(self, tensor):
        check = tensor.min()
        if check < 0:
            print(f'RH: check= {check}')
        if torch.rand(1) <= self.prob:
            if self.scaling == 'linear':
                scaler = torch.rand(1, device=tensor.device) * self.range + self.bounds[0]
                return torch.poisson(tensor * scaler) / scaler
            else:
                scaler = (((self.base**torch.rand(1, device=tensor.device) - 1)/(self.base-1)) * self.range) + self.bounds[0]
                return torch.poisson(tensor * scaler) / scaler
        else:
            return tensor
    
    def __repr__(self):
        return f"AddPoissonNoise(level_bounds={self.bounds}, prob={self.prob})"

class ScaleDynamicRange(Module):
    """
    Min-max scaling of the input tensor.
    RH 2021
    """
    def __init__(self, scaler_bounds=(0,1), epsilon=1e-9):
        """
        Initializes the class.
        Args:
            scaler_bounds (tuple):
                The bounds of how much to multiply the image by
                 prior to adding the Poisson noise.
             epsilon (float):
                 Value to add to the denominator when normalizing.
        """
        super().__init__()

        self.bounds = scaler_bounds
        self.range = scaler_bounds[1] - scaler_bounds[0]
        
        self.epsilon = epsilon
    
    def forward(self, tensor):
        tensor_minSub = tensor - tensor.min()
        return tensor_minSub * (self.range / (tensor_minSub.max(dim=-1,keepdim=True)[0].max(dim=-2,keepdim=True)[0]+self.epsilon))
    def __repr__(self):
        return f"ScaleDynamicRange(scaler_bounds={self.bounds})"

class Clip(Module):
    """
    Clips the input tensor to be between the lower and upper bounds.
    RH 2021
    """
    def __init__(self, lower_bound=0., upper_bound=1.):
        """
        Initializes the class.
        Args:
            lower_bound (float):
                The lower bound.
            upper_bound (float):
                The upper bound.
        """
        super().__init__()

        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def forward(self, tensor):
        return torch.clamp(tensor, min=self.lower_bound, max=self.upper_bound)
    def __repr__(self):
        return f"Clip(lower_bound={self.lower_bound}, upper_bound={self.upper_bound})"

class TileChannels(Module):
    """
    Expand dimension dim in X_in and tile to be N channels.
    RH 2021
    """
    def __init__(self, dim=0, n_channels=3):
        """
        Initializes the class.
        Args:
            dim (int):
                The dimension to tile.
            n_channels (int):
                The number of channels to tile to.
        """
        super().__init__()
        self.dim = dim
        self.n_channels = n_channels

    def forward(self, tensor):
        dims = [1]*len(tensor.shape)
        dims[self.dim] = self.n_channels
        return torch.tile(tensor, dims)
    def __repr__(self):
        return f"TileChannels(dim={self.dim})"

class Normalize(Module):
    """
    Normalizes the input tensor by setting the 
     mean and standard deviation of each channel.
    For imagenet, use means=[0.485, 0.456, 0.406] and
     stds=[0.229, 0.224, 0.225].
    RH 2021
    """
    def __init__(self, means=0, stds=1):
        """
        Initializes the class.
        Args:
            mean (float):
                Mean to set.
            std (float):
                Standard deviation to set.
        """
        super().__init__()
        self.means = torch.as_tensor(means)[:,None,None]
        self.stds = torch.as_tensor(stds)[:,None,None]
    def forward(self, tensor):
        tensor_means = tensor.mean(dim=(-2, -1), keepdim=True)
        tensor_stds = tensor.std(dim=(-2, -1), keepdim=True)
        tensor_z = (tensor - tensor_means) / tensor_stds
        return (tensor_z * self.stds) + self.means

class WarpPoints(Module):
    """
    Warps the input tensor at the given points by the given deltas.
    RH 2021 / JZ 2021
    """
    
    def __init__(self,  r=[0, 2],
                        cx=[-0.5, 0.5],
                        cy=[-0.5, 0.5], 
                        dx=[-0.3, 0.3], 
                        dy=[-0.3, 0.3], 
                        n_warps=1,
                        prob=0.5,
                        img_size_in=[36, 36],
                        img_size_out=[36, 36]):
        """
        Initializes the class.

        Args:
            r (list):
                The range of the radius.
            cx (list):
                The range of the center x.
            cy (list):  
                The range of the center y.
            dx (list):
                The range of the delta x.
            dy (list):
                The range of the delta y.
            n_warps (int):
                The number of warps to apply.
            prob (float):
                The probability of adding noise at all.
            img_size_in (list):
                The size of the input image.
            img_size_out (list):
                The size of the output image.
        """
        
        super().__init__()

        self.r = r
        self.cx = cx
        self.cy = cy
        self.dx = dx
        self.dy = dy
        self.n_warps = n_warps

        self.prob = prob

        self.img_size_in = img_size_in
        self.img_size_out = img_size_out

        self.r_range = r[1] - r[0]
        self.cx_range = cx[1] - cx[0]
        self.cy_range = cy[1] - cy[0]
        self.dx_range = dx[1] - dx[0]
        self.dy_range = dy[1] - dy[0]
        
        #### , indexing='ij' within torch.meshgrid call to remove warning
        
        self.meshgrid_in =  torch.tile(torch.stack(torch.meshgrid(torch.linspace(-1, 1, self.img_size_in[0]),  torch.linspace(-1, 1, self.img_size_in[1]), indexing='ij'), dim=0)[...,None], (1,1,1, n_warps))
        self.meshgrid_out = torch.tile(torch.stack(torch.meshgrid(torch.linspace(-1, 1, self.img_size_out[0]), torch.linspace(-1, 1, self.img_size_out[1]), indexing='ij'), dim=0)[...,None], (1,1,1, n_warps))
        

    def gaus2D(self, x, y, sigma):
        return torch.exp(-((torch.square(self.meshgrid_out[0] - x[None,None,:]) + torch.square(self.meshgrid_out[1] - y[None,None,:]))/(2*torch.square(sigma[None,None,:]))))        

    def forward(self, tensor):
        if tensor.ndim == 3:
            tensor = tensor[None, ...]
            flag_batched = False
        elif tensor.ndim == 4:
            flag_batched = True
        else:
            raise ValueError("Input tensor must be 3 or 4 dimensional.")
        
        if torch.rand(1) <= self.prob:
            rands = torch.rand(5, self.n_warps)
            cx = rands[0,:] * (self.cx_range) + self.cx[0]
            cy = rands[1,:] * (self.cy_range) + self.cy[0]
            dx = rands[2,:] * (self.dx_range) + self.dx[0]
            dy = rands[3,:] * (self.dy_range) + self.dy[0]
            r =  rands[4,:] * (self.r_range)  + self.r[0]
            im_gaus = self.gaus2D(x=cx, y=cy, sigma=r) # shape: (img_size_x, img_size_y, n_warps)
            im_disp = im_gaus[None,...] * torch.stack([dx, dy], dim=0).reshape(2, 1, 1, self.n_warps) # shape: (2(dx,dy), img_size_x, img_size_y, n_warps)
            im_disp_composite = torch.sum(im_disp, dim=3, keepdim=True) # shape: (2(dx,dy), img_size_x, img_size_y)
            im_newPos = self.meshgrid_out[...,0:1] + im_disp_composite
        else:
            im_newPos = self.meshgrid_out[...,0:1]
        
        im_newPos = torch.permute(im_newPos, [3,2,1,0]) # Requires 1/2 transpose because otherwise results are transposed from torchvision Resize
        if flag_batched:
            ## repmat for batch dimension
            im_newPos = torch.tile(im_newPos, (tensor.shape[0], 1, 1, 1))
        ret = torch.nn.functional.grid_sample( tensor, 
                                                im_newPos, 
                                                mode='bilinear',
                                                # mode='bicubic', 
                                                padding_mode='zeros', 
                                                align_corners=True)
        ret = ret[0] if not flag_batched else ret
        return ret
        
    def __repr__(self):
        return f"WarpPoints(r={self.r}, cx={self.cx}, cy={self.cy}, dx={self.dx}, dy={self.dy}, n_warps={self.n_warps}, prob={self.prob}, img_size_in={self.img_size_in}, img_size_out={self.img_size_out})"
    
    

class Horizontal_stripe_scale(Module):
    """
    Adds horizontal stripes. Can be used for for augmentation
     in images generated from raster scanning with bidirectional
     phased raster scanning.
    RH 2022
    """
    def __init__(self, alpha_min_max=(0.5,1), im_size=(36,36), prob=0.5):
        """
        Initializes the class.
        Args:
            alpha_min_max (2-tuple of floats):
                Range of scaling to apply to stripes.
                Will be pulled from uniform distribution.
        """
        super().__init__()
        
        self.alpha_min = alpha_min_max[0]
        self.alpha_max = alpha_min_max[1]
        self.alpha_range = alpha_min_max[1] - alpha_min_max[0]
        
        self.stripes_odd   = (torch.arange(im_size[0]) % 2)
        self.stripes_even = ((torch.arange(im_size[0])+1) % 2)
        
        self.prob = prob

    def forward(self, tensor):
#         assert tensor.ndim==3, "RH ERROR: Number of dimensions of input tensor should be 3: (n_images, height, width)"
        
        if torch.rand(1) < self.prob:
            n_ims = tensor.shape[0]
            alphas_odd  = (torch.rand(n_ims)*self.alpha_range) + self.alpha_min
            alphas_even = (torch.rand(n_ims)*self.alpha_range) + self.alpha_min

            stripes_mask = (self.stripes_odd[None,:]*alphas_odd[:,None]) + (self.stripes_even[None,:]*alphas_even[:,None])
            mask = torch.ones(tensor.shape[1], tensor.shape[2]) * stripes_mask[:,:,None]

            return mask*tensor
        else:
            return tensor

class Horizontal_stripe_shift(Module):
    """
    Shifts horizontal stripes. Can be used for for augmentation
     in images generated from raster scanning with bidirectional
     phased raster scanning.
    RH 2022
    """
    def __init__(self, alpha_min_max=(0,5), im_size=(36,36), prob=0.5):
        """
        Initializes the class.
        Args:
            alpha_min_max (2-tuple of ints):
                Range of absolute shift differences between
                 adjacent horizontal lines. INCLUSIVE.
                In pixels.
                Will be pulled from uniform distribution.
        """
        super().__init__()
        
        self.alpha_min = int(alpha_min_max[0])
        self.alpha_max = int(alpha_min_max[1] + 1)
        self.alpha_range = int(alpha_min_max[1] - alpha_min_max[0])
        
        self.idx_odd   = (torch.arange(im_size[0]) % 2).type(torch.bool)
        self.idx_even = ((torch.arange(im_size[0])+1) % 2).type(torch.bool)
        
        self.prob = prob
        
    def forward(self, tensor):
#         assert tensor.ndim==3, "RH ERROR: Number of dimensions of input tensor should be 3: (n_images, height, width)"
        
        if torch.rand(1) < self.prob:
            n_ims = tensor.shape[0]
            shape_im = (tensor.shape[1], tensor.shape[2])

            alpha = torch.randint(low=self.alpha_min, high=self.alpha_max, size=[n_ims]) * (torch.randint(low=0, high=2, size=[n_ims])*2 - 1)
#             alpha = (torch.randint(high=self.alpha_max-self.alpha_min, size=[n_ims]) + self.alpha_min) * (torch.randint(high=2, size=[n_ims])*2 - 1)
            alpha_half = alpha/2
            alphas_odd  =  torch.ceil(alpha_half).type(torch.int64)
            alphas_even = -torch.floor(alpha_half).type(torch.int64)

            out = torch.zeros_like(tensor)
            for ii in range(out.shape[0]):
                idx_take = slice(max(0, -alphas_odd[ii]) , min(shape_im[1], shape_im[1]-alphas_odd[ii]))
                idx_put = slice(max(0, alphas_odd[ii]) , min(shape_im[1], shape_im[1]+alphas_odd[ii]))
                out[ii, self.idx_odd, idx_put] = tensor[ii, self.idx_odd, idx_take]

                idx_take = slice(max(0, -alphas_even[ii]) , min(shape_im[1], shape_im[1]-alphas_even[ii]))
                idx_put = slice(max(0, alphas_even[ii]) , min(shape_im[1], shape_im[1]+alphas_even[ii]))
                out[ii, self.idx_even, idx_put] = tensor[ii, self.idx_even, idx_take]

            return out
        else:
            return tensor



class Scale_image_sum(Module):
    """
    Scales the entire image so that the sum is user-defined.
    RH 2022
    """
    def __init__(self, sum_val:float=1.0, epsilon=1e-9, min_sub=True):
        """
        Initializes the class.
        Args:
            sum_val (float):
                Value used to normalize the sum of each image.
            epsilon (float):
                Value added to denominator to prevent 
                 dividing by zero.
            min_sub (bool):
                If True, subtracts the minimum value of each
                 image so that the minimum value is zero.
        """
        super().__init__()
        
        self.sum_val=sum_val
        self.epsilon=epsilon
        self.min_sub=min_sub

    def forward(self, tensor):
        out = self.sum_val * (tensor / (torch.sum(tensor, dim=(-2,-1), keepdim=True) + self.epsilon))
        if self.min_sub:
            # out = out - torch.min(out.reshape(out.shape[0], out.shape[1], -1), dim=-1, keepdim=True)[0][...,None]
            out = out - torch.min(out.reshape(list(out.shape[:-2]) + [-1]), dim=-1, keepdim=True)[0][...,None]
        return out
    

class Check_NaN(Module):
    """
    Checks for NaNs.
    RH 2022
    """
    def __init__(self):
        super().__init__()
        

    def forward(self, tensor):
        if tensor.isnan().any():
            print('FOUND NaN')
            
        return tensor


class Random_occlusion(Module):
    """
    Randomly occludes a slice of the entire image.
    RH 2022
    """
    def __init__(self, prob=0.5, size=(0.3, 0.5)):
        """
        Initializes the class.
        Args:
            prob (float):
                Probability of occlusion.
            size (2-tuple of floats):
                Size of occlusion.
                In percent of image size.
                Will be pulled from uniform distribution.
        """
        super().__init__()
        
        self.prob = prob
        self.size = size

        self.rotator = torchvision.transforms.RandomRotation(
            (-180,180),
            # interpolation='nearest', 
            expand=False, 
            center=None, 
            fill=0,
        )
        
    def forward(self, tensor):
        if torch.rand(1) < self.prob:
            size_rand = torch.rand(1) * (self.size[1] - self.size[0]) + self.size[0]
            idx_rand = ((torch.ceil(tensor.shape[2] * (1-size_rand)).int().item()) , 0)
            mask = torch.ones_like(tensor)
            mask[:, :, idx_rand[0]:, :] = torch.zeros(1)

            out = tensor * self.rotator(mask).type(torch.bool)
            return out
        else:
            return tensor
        

class Make_square(torch.nn.Module):
    """
    Makes an image square by padding. Different padding methods can be used.
    """
    def __init__(self, pad_method='constant', pad_value=0):
        """
        Initializes the class.
        Args:
            pad_method (str):
                Method used for padding.
                Options: 'constant', 'edge', 'reflect', 'replicate'
            pad_value (float):
                Value used for padding if pad_method is 'constant'.
        """
        super().__init__()

        self.pad_method = pad_method
        self.pad_value = pad_value

    def forward(self, tensor):
        """
        Pads the input tensor to make it square.

        Args:
            tensor (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The padded tensor.
        """
        # Get the height and width of the tensor
        h, w = tensor.shape[2:]

        # Calculate the size difference
        diff = abs(h - w)

        # Calculate the padding
        if h <= w:
            padding = (0, 0, diff // 2, diff - diff // 2)
        else:
            padding = (diff // 2, diff - diff // 2, 0, 0)

        # Apply the padding
        return torch.nn.functional.pad(tensor, padding, mode=self.pad_method, value=self.pad_value)


class Random_multiply(torch.nn.Module):
    """
    Randomly multiplies the input tensor by a random number.
    """
    def __init__(self, min_magnitude=-1.0, max_magnitude=1.0, positive=True, negative=True, p=0.5):
        """
        Initializes the class.
        Args:
            min_magnitude (float):
                Minimum magnitude of random number.
                If a tuple, then length must be equal to the number of channels.
            max_magnitude (float):
                Maximum magnitude of random number.
                If a tuple, then length must be equal to the number of channels.
            positive (bool):
                If True, then the random values will be positive.
            negative (bool):
                If True, then the random values will be negative.
            p (float):
                Probability of multiplication.
        """
        super().__init__()

        if isinstance(min_magnitude, (tuple, list, torch.Tensor)) or isinstance(max_magnitude, (tuple, list, torch.Tensor)):
            assert len(min_magnitude) == len(max_magnitude), 'Length of min_magnitude and max_magnitude must be equal.'
        elif isinstance(min_magnitude, (int, float)):
            min_magnitude = [min_magnitude]
            max_magnitude = [max_magnitude]
        self.min_magnitude = torch.as_tensor(min_magnitude, dtype=torch.float32)
        self.max_magnitude = torch.as_tensor(max_magnitude, dtype=torch.float32)
        self.positive = positive
        self.negative = negative
        self.p = p
        self.n = len(self.min_magnitude)

        self.range = self.max_magnitude - self.min_magnitude

    def forward(self, tensor):
        """
        Randomly multiplies the input tensor by a random number.

        Args:
            tensor (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The multiplied tensor.
        """
        if torch.rand(1) < self.p:
            val = (torch.rand(self.n) * self.range + self.min_magnitude)
            if self.positive and self.negative:
                val = val * torch.sign(torch.rand(self.n) - 0.5)
            elif self.positive and not self.negative:
                val = torch.abs(val)
            elif self.negative and not self.positive:
                val = -torch.abs(val)
            return tensor * val[None, :, None, None]
        else:
            return tensor
        
class To_tensor(torch.nn.Module):
    """
    Converts PIL image to tensor.
    """
    def __init__(self):
        """
        Initializes the class.
        """
        super().__init__()

    def forward(self, tensor, output_ndim=4, *args, **kwargs):
        """
        Converts PIL image to tensor.

        Args:
            tensor (torch.Tensor): The input tensor.
            output_ndim (int): The number of dimensions of the output tensor.
            args: Additional positional arguments.
            kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: The multiplied tensor.
        """
        out = torchvision.transforms.functional.to_tensor(tensor, *args, **kwargs)
        if output_ndim == 4:
            out = out[None, ...]
        return out


class InverseShadow(torch.nn.Module):
    """
    Makes a drop shadow that is grayscale and the inverse sign of the input
    image.
    RH 2023
    """
    def __init__(
        self,
        blur_kernel_size=(20, 20),
        blur_sigma=(5.0, 5.0),
        p=0.5,
        normalize=True,
        norm_scale=(1.0, 10.0),
    ):
        """
        Initializes the class.
        """
        super().__init__()
        
        self.fn_blur = torchvision.transforms.GaussianBlur(kernel_size=blur_kernel_size, sigma=blur_sigma)

        self.p = p
        self.normalize = normalize
        self.norm_scale = norm_scale

    def forward(
        self,
        tensor,
    ):
        """
        Makes a drop shadow that is grayscale and the inverse sign of the input.

        Args:
            tensor (torch.Tensor): 
                The input tensor.
                Shape: (N, C, H, W)
        """
        if torch.rand(1) < self.p:
            ## Make grayscale
            im = tensor.mean(1, keepdim=True)
            ## Blur and invert
            im = self.fn_blur(im) * -1
            ## Normalize
            if self.normalize:
                scale = torch.rand(1) * (self.norm_scale[1] - self.norm_scale[0]) + self.norm_scale[0]
                im = im / (torch.abs(tensor.mean() / scale))
            ## Apply
            return tensor + im
        else:
            return tensor



##############################################################################################################
################################################ KORNIA STUFF ################################################
##############################################################################################################

import kornia

class Kornia_filter_to_augmentation(torch.nn.Module):
    """
    Converts a Kornia filter to an augmentation. Provides probability and
    magnitude control.

    Args:
        filter (torch.nn.Module): 
            The Kornia filter.
        prob (float): 
            The probability of applying the filter.
        kwargs: 
            Keyword arguments for the filter. The keys are the names of the
            filter parameters. If the value is a list type, then it should be of the
            form (min, max), and the value used for the kwarg will be randomly
            sampled from the range (min, max).
    """
    def __init__(self, filter, prob=0.5, **kwargs):
        super().__init__()

        self.filter = filter
        self.prob = prob
        self.kwargs = kwargs

    def forward(self, tensor):
        if torch.rand(1) < self.prob:
            kwargs = {}
            for k, v in self.kwargs.items():
                if isinstance(v, (list)):
                    kwargs[k] = torch.rand(1) * (v[1] - v[0]) + v[0]
                else:
                    kwargs[k] = v
            return self.filter(tensor, **kwargs)
        else:
            return tensor

class RandomPlasmaMultiplication(torch.nn.Module):
    def __init__(
        self,
        roughness=(0.1, 0.7),
        intensity=(0.0, 1.0),
        p=0.5,
        same_on_batch=False,
        keepdim=False,
    ):
        super().__init__()
        self.rpb = kornia.augmentation.RandomPlasmaBrightness(roughness=roughness, intensity=intensity, p=p, same_on_batch=same_on_batch, keepdim=keepdim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ones = torch.ones_like(x)
        plasma = self.rpb(ones)
        return x * plasma
