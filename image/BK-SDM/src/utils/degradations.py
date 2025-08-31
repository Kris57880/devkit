import numpy as np
import torch
from omegaconf import DictConfig
import os
# from .jpeg_torch import jpeg_decode, jpeg_encode

import yaml
from torch.nn import functional as F
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import subprocess
import compressai 
import logging
import contextlib
import sys
import time
from compressai.zoo import load_state_dict
from . import elic
from .elic.Network import TestModel
import random

# from utils.fft_utils import fft2_m, ifft2_m   

def set_all_seeds(seed=1024):
    """固定所有可能的隨機性來源"""
    # Python 隨機模組
    random.seed(seed)
    
    # NumPy 隨機性
    np.random.seed(seed)
    
    # PyTorch 隨機性
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 多 GPU 支持
    
    # CuDNN 確定性設置
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # PyTorch 其他設置
    torch.use_deterministic_algorithms(True)
    
    # 設置環境變數以確保完全確定性
    os.environ['PYTHONHASHSEED'] = str(seed)

class H_functions:
    """
    A class replacing the SVD of a matrix H, perhaps efficiently.
    All input vectors are of shape (Batch, ...).
    All output vectors are of shape (Batch, DataDimension).
    """

    def V(self, vec):
        """
        Multiplies the input vector by V
        """
        raise NotImplementedError()

    def Vt(self, vec):
        """
        Multiplies the input vector by V transposed
        """
        raise NotImplementedError()

    def U(self, vec):
        """
        Multiplies the input vector by U
        """
        raise NotImplementedError()

    def Ut(self, vec):
        """
        Multiplies the input vector by U transposed
        """
        raise NotImplementedError()

    def singulars(self):
        """
        Returns a vector containing the singular values. The shape of the vector should be the same as the smaller dimension (like U)
        """
        raise NotImplementedError()

    def add_zeros(self, vec):
        """
        Adds trailing zeros to turn a vector from the small dimension (U) to the big dimension (V)
        """
        raise NotImplementedError()

    def H(self, vec):
        """
        Multiplies the input vector by H
        """
        temp = self.Vt(vec)
        singulars = self.singulars()
        return self.U(singulars * temp[:, : singulars.shape[0]])

    def Ht(self, vec):
        """
        Multiplies the input vector by H transposed
        """
        temp = self.Ut(vec)
        singulars = self.singulars()
        return self.V(self.add_zeros(singulars * temp[:, : singulars.shape[0]]))

    def H_pinv(self, vec):
        """
        Multiplies the input vector by the pseudo inverse of H
        """
        temp = self.Ut(vec)  # (b, m) - > (b, m)
        singulars = self.singulars()  # (mxm, )
        # temp[:, :singulars.shape[0]] = temp[:, :singulars.shape[0]] / singulars
        nonzero_idx = singulars.nonzero().flatten()
        temp[:, nonzero_idx] = temp[:, nonzero_idx] / singulars[nonzero_idx]

        return self.V(self.add_zeros(temp))


# a memory inefficient implementation for any general degradation H
class GeneralH(H_functions):
    def mat_by_vec(self, M, v):
        vshape = v.shape[1]
        if len(v.shape) > 2:
            vshape = vshape * v.shape[2]
        if len(v.shape) > 3:
            vshape = vshape * v.shape[3]
        return torch.matmul(M, v.view(v.shape[0], vshape, 1)).view(v.shape[0], M.shape[0])

    def __init__(self, H):
        self._U, self._singulars, self._V = torch.svd(H, some=False)
        self._Vt = self._V.transpose(0, 1)
        self._Ut = self._U.transpose(0, 1)

        ZERO = 1e-3
        self._singulars[self._singulars < ZERO] = 0
        print(len([x.item() for x in self._singulars if x == 0]))

    def V(self, vec):
        return self.mat_by_vec(self._V, vec.clone())

    def Vt(self, vec):
        return self.mat_by_vec(self._Vt, vec.clone())

    def U(self, vec):
        return self.mat_by_vec(self._U, vec.clone())

    def Ut(self, vec):
        return self.mat_by_vec(self._Ut, vec.clone())

    def singulars(self):
        return self._singulars

    def add_zeros(self, vec):
        out = torch.zeros(vec.shape[0], self._V.shape[0], device=vec.device)
        out[:, : self._U.shape[0]] = vec.clone().reshape(vec.shape[0], -1)
        return out



class SparseH(GeneralH):
    def __init__(self, U, S, V):
        self._U = U.to_sparse()
        self._V = V.to_sparse()
        self._singulars = S
        ZERO = 1e-6
        self._singulars[self._singulars < ZERO] = 0

    def mat_by_vec(self, M, v):
        vshape = v.shape[1]
        if len(v.shape) > 2:
            vshape = vshape * v.shape[2]
        if len(v.shape) > 3:
            vshape = vshape * v.shape[3]

        return torch.sparse.mm(M, v.view(v.shape[0], vshape, 1)).view(v.shape[0], M.shape[0])


class Inpainting2(H_functions):
    def __init__(self, channels, img_dim, dense_masks, device):
        self.channels = channels
        self.img_dim = img_dim
        self.dense_masks = dense_masks
        self.device = device

    def set_indices(self, idx):
        idx = torch.remainder(idx, self.dense_masks.size(0))
        _singulars = self.dense_masks[idx].clone().to(self.device)
        self._singulars = _singulars.reshape(_singulars.size(0), -1)

    def V(self, vec):
        return vec.reshape(vec.size(0), -1)

    def Vt(self, vec):
        return vec.reshape(vec.size(0), -1)

    def U(self, vec):
        return vec.reshape(vec.size(0), -1)

    def Ut(self, vec):
        return vec.reshape(vec.size(0), -1)

    def add_zeros(self, vec):
        return vec

    def singulars(self):
        return self._singulars.float()

    def H(self, vec):
        return vec.reshape(*self.singulars().size()) * self.singulars()

    def H_pinv(self, vec):
        return vec.reshape(*self.singulars().size()) * self.singulars()

# Inpainting
class Inpainting(H_functions):
    def __init__(self, channels, img_dim, dense_masks, device):
        self.channels = channels
        self.img_dim = img_dim
        self.dense_masks = dense_masks
        n = dense_masks.size(0)
        
            
        # xxx = [torch.nonzero(dense_masks[i] == 0).long() for i in range(n)]
        # print('xxx', len(xxx))
        # for i in range(len(xxx)):
        #     print('i', i)
        #     print(xxx[i].shape)

        # import pdb; pdb.set_trace()
        
        #self.missing_masks = torch.cat([torch.nonzero(dense_masks[i] == 0).long() for i in range(n)], dim=0).T
        # self.keep_masks = torch.cat([torch.nonzero(dense_masks[i] != 0).long() for i in range(n)], dim=0).T
        
        self.missing_masks = torch.cat([torch.nonzero(dense_masks[i] == 0).long() for i in range(n)], dim=1).T
        self.keep_masks = torch.cat([torch.nonzero(dense_masks[i] != 0).long() for i in range(n)], dim=1).T
        
        self.device = device
        
    def set_indices(self, idx):
        channels = self.channels
        img_dim = self.img_dim
        device = self.device
        idx = torch.remainder(idx, self.missing_masks.size(0))
        missing_masks = self.missing_masks[idx].clone().to(device)
        # ###
        # l = missing_masks.size(1)im
        # missing_masks = torch.div(missing_masks[:l//3], 4, rounding_mode="floor").unique()
        # missing_masks = torch.cat([missing_masks, missing_masks + channels * img_dim * img_dim // 4, missing_masks + channels * img_dim * img_dim // 4 * 2])
        # missing_masks = missing_masks.view(1, -1)
        # ###
        missing_masks = missing_masks
        print('missing_masks.shape', missing_masks.shape)
        self.missing_indices = missing_indices = missing_masks # self.missing_masks[idx].clone().to(device)
        self._singulars = torch.ones(channels * img_dim ** 2 - missing_indices.shape[1]).to(device)
        kept_masks = self.keep_masks[idx].clone().to(device)
        # l = kept_masks.size(1)
        # kept_masks = torch.div(kept_masks[:l//3], 4, rounding_mode="floor").unique()
        # kept_masks = torch.cat([kept_masks, kept_masks + channels * img_dim * img_dim // 4, kept_masks + channels * img_dim * img_dim // 4 * 2])
        # kept_masks = kept_masks.view(1, -1)
        self.kept_indices = kept_masks # self.keep_masks[idx].clone().to(device)

    def V(self, vec):
        temp = vec.clone().reshape(vec.shape[0], -1)
        out = torch.zeros_like(temp)
        assert vec.size(0) == self.kept_indices.size(0)
        n = vec.size(0)
        for i in range(n):
            out[i, self.kept_indices[i]] = temp[i, : self.kept_indices[i].shape[0]]
            out[i, self.missing_indices[i]] = temp[i, self.kept_indices[i].shape[0] :]
        return out #.reshape(vec.shape[0], -1, self.channels).permute(0, 2, 1).reshape(vec.shape[0], -1)

    def Vt(self, vec):
        temp = vec.clone().reshape(vec.shape[0], -1)
        out = torch.zeros_like(temp)
        assert vec.size(0) == self.kept_indices.size(0)
        # import ipdb; ipdb.set_trace()
        n = vec.size(0)
        for i in range(n):
            out[i, : self.kept_indices[i].shape[0]] = temp[i, self.kept_indices[i]]
            out[i, self.kept_indices[i].shape[0] :] = temp[i, self.missing_indices[i]]
        return out

    def U(self, vec):
        return vec.clone().reshape(vec.shape[0], -1)

    def Ut(self, vec):
        return vec.clone().reshape(vec.shape[0], -1)

    def singulars(self):
        return self._singulars

    def add_zeros(self, vec):
        temp = torch.zeros((vec.shape[0], self.channels * self.img_dim ** 2), device=vec.device)
        reshaped = vec.clone().reshape(vec.shape[0], -1)
        temp[:, : reshaped.shape[1]] = reshaped
        return temp

    def H(self, vec):
        """
        Multiplies the input vector by H
        """
        temp = self.Vt(vec)
        singulars = self.singulars()
        return self.U(singulars * temp[:, :singulars.shape[0]])

    def Ht(self, vec):
        """
        Multiplies the input vector by H transposed
        """
        temp = self.Ut(vec)
        singulars = self.singulars()
        return self.V(self.add_zeros(singulars * temp[:, : singulars.shape[0]]))

    def H_pinv(self, vec):
        """
        Multiplies the input vector by the pseudo inverse of H
        """
        temp = self.Ut(vec)  # (b, m) - > (b, m)
        singulars = self.singulars()  # (mxm, )
        # temp[:, :singulars.shape[0]] = temp[:, :singulars.shape[0]] / singulars
        nonzero_idx = singulars.nonzero().flatten()
        temp[:, nonzero_idx] = temp[:, nonzero_idx] / singulars[nonzero_idx]

        return self.V(self.add_zeros(temp))




# Denoising
class Denoising(H_functions):
    def __init__(self, channels, img_dim, device):
        self._singulars = torch.ones(channels * img_dim ** 2, device=device)

    def V(self, vec):
        return vec.clone().reshape(vec.shape[0], -1)

    def Vt(self, vec):
        return vec.clone().reshape(vec.shape[0], -1)

    def U(self, vec):
        return vec.clone().reshape(vec.shape[0], -1)

    def Ut(self, vec):
        return vec.clone().reshape(vec.shape[0], -1)

    def singulars(self):
        return self._singulars

    def add_zeros(self, vec):
        return vec.clone().reshape(vec.shape[0], -1)


# Super Resolution
class SuperResolution(H_functions):
    def __init__(self, channels, img_dim, ratio, device):  # ratio = 2 or 4
        assert img_dim % ratio == 0
        self.img_dim = img_dim
        self.channels = channels
        self.y_dim = img_dim // ratio
        self.ratio = ratio
        H = torch.Tensor([[1 / ratio ** 2] * ratio ** 2]).to(device)
        self.U_small, self.singulars_small, self.V_small = torch.svd(H, some=False)
        self.Vt_small = self.V_small.transpose(0, 1)

    def V(self, vec):
        # reorder the vector back into patches (because singulars are ordered descendingly)
        temp = vec.clone().reshape(vec.shape[0], -1)
        patches = torch.zeros(vec.shape[0], self.channels, self.y_dim ** 2, self.ratio ** 2, device=vec.device,)
        patches[:, :, :, 0] = temp[:, : self.channels * self.y_dim ** 2].view(vec.shape[0], self.channels, -1)
        for idx in range(self.ratio ** 2 - 1):
            patches[:, :, :, idx + 1] = temp[:, (self.channels * self.y_dim ** 2 + idx) :: self.ratio ** 2 - 1].view(
                vec.shape[0], self.channels, -1
            )
        # multiply each patch by the small V
        patches = torch.matmul(self.V_small, patches.reshape(-1, self.ratio ** 2, 1)).reshape(
            vec.shape[0], self.channels, -1, self.ratio ** 2
        )
        # repatch the patches into an image
        patches_orig = patches.reshape(vec.shape[0], self.channels, self.y_dim, self.y_dim, self.ratio, self.ratio)
        recon = patches_orig.permute(0, 1, 2, 4, 3, 5).contiguous()
        recon = recon.reshape(vec.shape[0], self.channels * self.img_dim ** 2)
        return recon

    def Vt(self, vec):
        # extract flattened patches
        patches = vec.clone().reshape(vec.shape[0], self.channels, self.img_dim, self.img_dim)
        patches = patches.unfold(2, self.ratio, self.ratio).unfold(3, self.ratio, self.ratio)
        unfold_shape = patches.shape
        patches = patches.contiguous().reshape(vec.shape[0], self.channels, -1, self.ratio ** 2)
        # multiply each by the small V transposed
        patches = torch.matmul(self.Vt_small, patches.reshape(-1, self.ratio ** 2, 1)).reshape(
            vec.shape[0], self.channels, -1, self.ratio ** 2
        )
        # reorder the vector to have the first entry first (because singulars are ordered descendingly)
        recon = torch.zeros(vec.shape[0], self.channels * self.img_dim ** 2, device=vec.device)
        recon[:, : self.channels * self.y_dim ** 2] = patches[:, :, :, 0].view(
            vec.shape[0], self.channels * self.y_dim ** 2
        )
        for idx in range(self.ratio ** 2 - 1):
            recon[:, (self.channels * self.y_dim ** 2 + idx) :: self.ratio ** 2 - 1] = patches[:, :, :, idx + 1].view(
                vec.shape[0], self.channels * self.y_dim ** 2
            )
        return recon

    def U(self, vec):
        return self.U_small[0, 0] * vec.clone().reshape(vec.shape[0], -1)

    def Ut(self, vec):  # U is 1x1, so U^T = U
        return self.U_small[0, 0] * vec.clone().reshape(vec.shape[0], -1)

    def singulars(self):
        return self.singulars_small.repeat(self.channels * self.y_dim ** 2)

    def add_zeros(self, vec):
        reshaped = vec.clone().reshape(vec.shape[0], -1)
        temp = torch.zeros((vec.shape[0], reshaped.shape[1] * self.ratio ** 2), device=vec.device)
        temp[:, : reshaped.shape[1]] = reshaped
        return temp


# Colorization
class Colorization(H_functions):
    def __init__(self, img_dim, device):
        self.channels = 3
        self.img_dim = img_dim
        # Do the SVD for the per-pixel matrix
        H = torch.Tensor([[0.3333, 0.3334, 0.3333]]).to(device)
        self.U_small, self.singulars_small, self.V_small = torch.svd(H, some=False)
        self.Vt_small = self.V_small.transpose(0, 1)

    def V(self, vec):
        # get the needles
        needles = vec.clone().reshape(vec.shape[0], self.channels, -1).permute(0, 2, 1)  # shape: B, WH, C'
        # multiply each needle by the small V
        needles = torch.matmul(self.V_small, needles.reshape(-1, self.channels, 1)).reshape(
            vec.shape[0], -1, self.channels
        )  # shape: B, WH, C
        # permute back to vector representation
        recon = needles.permute(0, 2, 1)  # shape: B, C, WH
        return recon.reshape(vec.shape[0], -1)

    def Vt(self, vec):
        # get the needles
        needles = vec.clone().reshape(vec.shape[0], self.channels, -1).permute(0, 2, 1)  # shape: B, WH, C
        # multiply each needle by the small V transposed
        needles = torch.matmul(self.Vt_small, needles.reshape(-1, self.channels, 1)).reshape(
            vec.shape[0], -1, self.channels
        )  # shape: B, WH, C'
        # reorder the vector so that the first entry of each needle is at the top
        recon = needles.permute(0, 2, 1).reshape(vec.shape[0], -1)
        return recon

    def U(self, vec):
        return self.U_small[0, 0] * vec.clone().reshape(vec.shape[0], -1)

    def Ut(self, vec):  # U is 1x1, so U^T = U
        return self.U_small[0, 0] * vec.clone().reshape(vec.shape[0], -1)

    def singulars(self):
        return self.singulars_small.repeat(self.img_dim ** 2)

    def add_zeros(self, vec):
        reshaped = vec.clone().reshape(vec.shape[0], -1)
        temp = torch.zeros((vec.shape[0], self.channels * self.img_dim ** 2), device=vec.device)
        temp[:, : self.img_dim ** 2] = reshaped
        return temp

# Deblurring
class Deblurring(H_functions):
    def mat_by_img(self, M, v):
        return torch.matmul(M, v.reshape(v.shape[0] * self.channels, self.img_dim, self.img_dim)).reshape(
            v.shape[0], self.channels, M.shape[0], self.img_dim
        )

    def img_by_mat(self, v, M):
        return torch.matmul(v.reshape(v.shape[0] * self.channels, self.img_dim, self.img_dim), M).reshape(
            v.shape[0], self.channels, self.img_dim, M.shape[1]
        )

    def __init__(self, kernel, channels, img_dim, device):
        self.img_dim = img_dim
        self.channels = channels
        # build 1D conv matrix
        H_small = torch.zeros(img_dim, img_dim, device=device)
        for i in range(img_dim):
            for j in range(i - kernel.shape[0] // 2, i + kernel.shape[0] // 2):
                if j < 0 or j >= img_dim:
                    continue
                H_small[i, j] = kernel[j - i + kernel.shape[0] // 2]
        # get the svd of the 1D conv
        self.U_small, self.singulars_small, self.V_small = torch.svd(H_small, some=False)
        ZERO = 3e-2
        self.singulars_small[self.singulars_small < ZERO] = 0
        # calculate the singular values of the big matrix
        self._singulars = torch.matmul(
            self.singulars_small.reshape(img_dim, 1), self.singulars_small.reshape(1, img_dim),
        ).reshape(img_dim ** 2)
        # sort the big matrix singulars and save the permutation
        self._singulars, self._perm = self._singulars.sort(descending=True)  # , stable=True)

    def V(self, vec):
        # invert the permutation
        temp = torch.zeros(vec.shape[0], self.img_dim ** 2, self.channels, device=vec.device)
        temp[:, self._perm, :] = vec.clone().reshape(vec.shape[0], self.img_dim ** 2, self.channels)
        temp = temp.permute(0, 2, 1)
        # multiply the image by V from the left and by V^T from the right
        out = self.mat_by_img(self.V_small, temp)
        out = self.img_by_mat(out, self.V_small.transpose(0, 1)).reshape(vec.shape[0], -1)
        return out

    def Vt(self, vec):
        # multiply the image by V^T from the left and by V from the right
        temp = self.mat_by_img(self.V_small.transpose(0, 1), vec.clone())
        temp = self.img_by_mat(temp, self.V_small).reshape(vec.shape[0], self.channels, -1)
        # permute the entries according to the singular values
        temp = temp[:, :, self._perm].permute(0, 2, 1)
        return temp.reshape(vec.shape[0], -1)

    def U(self, vec):
        # invert the permutation
        temp = torch.zeros(vec.shape[0], self.img_dim ** 2, self.channels, device=vec.device)
        temp[:, self._perm, :] = vec.clone().reshape(vec.shape[0], self.img_dim ** 2, self.channels)
        temp = temp.permute(0, 2, 1)
        # multiply the image by U from the left and by U^T from the right
        out = self.mat_by_img(self.U_small, temp)
        out = self.img_by_mat(out, self.U_small.transpose(0, 1)).reshape(vec.shape[0], -1)
        return out

    def Ut(self, vec):
        # multiply the image by U^T from the left and by U from the right
        temp = self.mat_by_img(self.U_small.transpose(0, 1), vec.clone())
        temp = self.img_by_mat(temp, self.U_small).reshape(vec.shape[0], self.channels, -1)
        # permute the entries according to the singular values
        temp = temp[:, :, self._perm].permute(0, 2, 1)
        return temp.reshape(vec.shape[0], -1)

    def singulars(self):
        return self._singulars.repeat(1, 3).reshape(-1)

    def add_zeros(self, vec):
        return vec.clone().reshape(vec.shape[0], -1)


class BDeblur(Deblurring):
    def H_pinv(self, vec):
        return vec   


class JPEG(H_functions):
    def __init__(self, qf):
        self.qf = qf

    def H(self, image):
        """
        Here, we write H as decode(encode()) simply because the resulting y is easier to visualize. 
        """
        return jpeg_decode(jpeg_encode(image, self.qf), self.qf)

    def H_pinv(self, x):
        return x

class MBT_2018(H_functions):
    def __init__(self, qf):
        codec_metric = 'mse'
        self.codec = compressai.zoo.mbt2018_mean(quality=qf, metric=codec_metric, pretrained=True, progress=True).cuda().eval()
        
    def H(self, image):
        image = (image +1)/2
        output = self.codec.forward(image)
        output['x_hat'].clamp_(0, 1)
        return (output['x_hat']*2)-1
    
    def H_pinv(self, x):
        return x    
class ELIC(H_functions):
    def __init__(self, qf, model_path=None, patch_size=64):
        """
        Initialize ELIC compression model
        Args:
            qf: quality factor (1, 2, 3, 4 corresponding to different compression levels)
            model_path: path to the ELIC model checkpoint (optional, will be auto-selected based on qf)
            patch_size: padding patch size for compression
        """
        self.patch_size = patch_size
        self.model = None
        self.qf = qf
        
        # Auto-select model path based on qf if not provided
        if model_path is None:
            model_path = self._get_model_path_by_qf(qf)
        
        self.model_path = model_path
        self._load_model()
        self._model_loaded = True
    
    def _get_model_path_by_qf(self, qf):
        """
        Get model path based on quality factor
        Args:
            qf: quality factor (1-4, where 1 is lowest quality/highest compression)
        Returns:
            path to the corresponding ELIC model
        """
        qf_to_model = {
            1: "ELIC_0004_ft_3980_Plateau.pth.tar",  
            2: "ELIC_0008_ft_3980_Plateau.pth.tar",
            3: "ELIC_0016_ft_3980_Plateau.pth.tar", 
            4: "ELIC_0032_ft_3980_Plateau.pth.tar"   
        }
        
        if qf not in qf_to_model:
            raise ValueError(f"Invalid quality factor {qf}. Must be one of {list(qf_to_model.keys())}")
        
        # Get the directory of this file
        current_dir = os.path.dirname(__file__)
        model_filename = qf_to_model[qf]
        model_path = os.path.join(current_dir, 'elic', 'checkpoints', model_filename)
        
        return model_path
        
    def _load_model(self):
                
        
        # Validate model path
        if self.model_path is None:
            raise ValueError("model_path must be provided for ELIC")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model checkpoint not found: {self.model_path}")
        
        # Load the model
        state_dict = load_state_dict(torch.load(self.model_path, map_location='cpu'))
        self.model = TestModel().cuda().eval()
        self.model.load_state_dict(state_dict)
        
        # Set entropy coder
        compressai.set_entropy_coder('ans')
        self._model_loaded = True
        # self.model=self.model.half()
    
    def decode(self, bitstream, h, w):
        """
        Decode an image using ELIC decompression
        """

        # Apply padding for compression
        # bitstream["strings"] = bitstream["strings"].half() if bitstream["strings"].dtype != torch.float16 else bitstream["strings"]
        # bitstream["shape"] = bitstream["shape"].half() if bitstream["shape"].dtype != torch.float16 else bitstream["shape"]
        # bitstream["strings"] = bitstream["strings"].to("cuda")
        # bitstream["shape"] = bitstream["shape"].to("cuda")
        out_dec = self.model.decompress(bitstream["strings"], bitstream["shape"])

        p = self.patch_size
        new_h = (h + p - 1) // p * p
        new_w = (w + p - 1) // p * p
        padding_left = 0
        padding_right = new_w - w - padding_left
        padding_top = 0
        padding_bottom = new_h - h - padding_top

        out_dec["x_hat"] = torch.nn.functional.pad(
            out_dec["x_hat"], (-padding_left, -padding_right, -padding_top, -padding_bottom)
        )
        return out_dec["x_hat"]
    
    


    def H(self, image):
        set_all_seeds(1024)
        """
        Apply ELIC compression and decompression
        Args:
            image: torch tensor with range [-1, 1] and shape (B, C, H, W)
        Returns:
            compressed and decompressed image with range [-1, 1]
        """        
        # Convert from [-1, 1] to [0, 1] range
        image_01 = (image + 1) / 2
        
        # Get original shape
        b, c, h, w = image_01.shape
        
        # Apply padding for compression
        p = self.patch_size
        new_h = (h + p - 1) // p * p
        new_w = (w + p - 1) // p * p
        padding_left = 0
        padding_right = new_w - w - padding_left
        padding_top = 0
        padding_bottom = new_h - h - padding_top
        
        # Pad the image
        pad = torch.nn.ConstantPad2d((padding_left, padding_right, padding_top, padding_bottom), 0)
        x_padded = pad(image_01)
        
        # Compress and decompress (with gradients enabled)
        # Note: The original inference function uses @torch.no_grad(), but we need gradients
        # so we call the model methods directly
        self.model = self.model.half()
        x_padded = x_padded.half()  # Convert to half precision
        # out_enc = self.model.compress(x_padded)
        # out_dec = self.model.decompress(out_enc["strings"], out_enc["shape"])
        out = self.model(x_padded)
        # Remove padding
        x_hat = torch.nn.functional.pad(
            out['x_hat'] ,
            (-padding_left, -padding_right, -padding_top, -padding_bottom)
        )
        # Clamp to valid range and convert back to [-1, 1]
        x_hat = torch.clamp(x_hat, 0, 1)
        result = x_hat * 2 - 1
        return result
    
    def H_pinv(self, x):
        """
        Pseudo-inverse operation for ELIC (identity function since compression is irreversible)
        """
        return x 




class MS_ILLM(H_functions):
    def __init__(self, qf):
        # Bitrate	MS-ILLM	No-GAN
        # 0.002 bpp	"msillm_quality_vlo1"	
        # 0.004 bpp	"msillm_quality_vlo2"	
        # 0.035 bpp	"msillm_quality_1"	"noganms_quality_1"
        # 0.07 bpp	"msillm_quality_2"	"noganms_quality_2"
        # 0.14 bpp	"msillm_quality_3"	"noganms_quality_3"
        # 0.3 bpp	"msillm_quality_4"	"noganms_quality_4"
        # 0.45 bpp	"msillm_quality_5"	"noganms_quality_5"
        # 0.9 bpp	"msillm_quality_6"	"noganms_quality_6"
        self.model = torch.hub.load("facebookresearch/NeuralCompression", f"msillm_quality_{qf}").cuda().eval()
        self.model.update()
        self.model.update_tensor_devices("compress")
    
    def H(self, image):
        image = (image +1)/2
        dtype = image.dtype
        if dtype != torch.float32:
            image = image.to(image.device, torch.float32)
        with torch.no_grad():
            compressed = self.model.compress(image, force_cpu=False)
            decompressed = self.model.decompress(compressed, force_cpu=False).clamp(0.0, 1.0)
        if dtype != torch.float32:
            decompressed = decompressed.to(dtype)
        return  decompressed * 2 - 1
        
    def H_pinv(self, x):
        return x
    
class Hyperprior(H_functions):
    def __init__(self, qf):
        codec_metric = 'mse'
        self.codec = compressai.zoo.bmshj2018_hyperprior(quality=qf, metric=codec_metric, pretrained=True, progress=True).cuda().eval()
        
    def H(self, image):
        image = (image +1)/2
        output = self.codec.forward(image)
        output['x_hat'].clamp_(0, 1)
        return (output['x_hat']*2)-1
    
    def H_pinv(self, x):
        return x    


class VVC(H_functions):
    def __init__(self, qp, vvc_root):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.qp = qp
        self.vvc_root = vvc_root

    def call_cmd(self, cmd, output_log=None):
        # print(f'cwd: {self.vvc_root}')
        # print(f'call: {cmd}')
        cwd = self.vvc_root
        try:
            if output_log:
                with open(output_log, "a") if output_log else None as log_file:
                    result = subprocess.run(cmd, shell=True, check=True, text=True, cwd=cwd,
                                            stdout=log_file, 
                                            stderr=log_file)
            else:
                result = subprocess.run(cmd, shell=True, check=True, text=True, cwd=cwd,
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE)
            # print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"Command failed: {cmd}")
            print(f"Error: {e.stderr}")
            assert False
    
    def png_to_yuv(self, png_path, yuv_path):
        cmd = f'ffmpeg -i {png_path} -pix_fmt yuv444p {yuv_path}'
        self.call_cmd(cmd)

    def intra_compress(self, in_yuv_path, out_yuv_path, w=256, h=256):
        cmd = f'bin/umake/gcc-11.4/x86_64/release/EncoderApp -c cfg/encoder_intra_vtm.cfg \
        -i {in_yuv_path} -b {out_yuv_path[:-4]}.bin -o {out_yuv_path} \
        -q {self.qp} -wdt {w} -hgt {h} -f 1 -fr 1 \
        --InputBitDepth=8 \
        --InputChromaFormat=444 \
        --ConformanceWindowMode=1'
        self.call_cmd(cmd, output_log=None)
        if os.path.exists(f'{out_yuv_path[:-4]}.bin'): os.remove(f'{out_yuv_path[:-4]}.bin')

    def yuv_to_png(self, yuv_path, png_path, w=256, h=256):
        cmd = f'ffmpeg -f rawvideo -s {w}x{h} -pixel_format yuv444p10le -i {yuv_path} -vframes 1 {png_path}'
        self.call_cmd(cmd)

    def H(self, image_tensor):
        # total_start = time.time()
        image_tensor = (image_tensor + 1) / 2
        vvc_imgs = []
        root = self.vvc_root
        h, w = image_tensor.size(2), image_tensor.size(3)
        
        # total_io_time = 0
        # total_compress_time = 0
        
        for i in range(image_tensor.size(0)):
            tmp_png = f"{root}/tmp.png"
            tmp_yuv = f"{root}/tmp.yuv"
            out_yuv = f"{root}/vvc_out.yuv"
            out_png = f"{root}/vvc_out.png"
            
            # IO operations timing
            # io_start = time.time()
            for file in [tmp_png, tmp_yuv, out_yuv, out_png]:
                if os.path.exists(file): os.remove(file) 

            torchvision.utils.save_image(image_tensor[i], tmp_png)
            self.png_to_yuv(tmp_png, tmp_yuv)
            # io_time_before_compress = time.time() - io_start
            
            # Compression timing
            # compress_start = time.time()
            self.intra_compress(tmp_yuv, out_yuv, w, h)
            # compress_time = time.time() - compress_start
            
            # IO operations after compression
            # io_start2 = time.time()
            self.yuv_to_png(out_yuv, out_png, w,h)
            img = Image.open(out_png).convert("RGB")
            vvc_img = self.transform(img).unsqueeze(0).to(image_tensor)
            vvc_imgs.append(vvc_img)
            # io_time_after_compress = time.time() - io_start2
            
            # img_io_time = io_time_before_compress + io_time_after_compress
            # total_io_time += img_io_time
            # total_compress_time += compress_time

            # for file in [tmp_png, tmp_yuv, out_yuv, out_png]:
            #     if os.path.exists(file): os.remove(file)        
        
        # total_time = time.time() - total_start
        # print(f"VVC H function - Total IO: {total_io_time:.3f}s ({total_io_time/total_time*100:.1f}%), " +
        #       f"Compress: {total_compress_time:.3f}s ({total_compress_time/total_time*100:.1f}%), " +
        #       f"Total: {total_time:.3f}s")
        
        return torch.cat(vvc_imgs, dim=0)

    def H_pinv(self, x):
        return x
    

class HEVC(H_functions):
    def __init__(self, qp):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.qp = qp

    def call_cmd(self, cmd, output_log=None):
        cwd = None
        try:
            if output_log:
                with open(output_log, "a") if output_log else None as log_file:
                    result = subprocess.run(cmd, shell=True, check=True, text=True, cwd=cwd,
                                            stdout=log_file, 
                                            stderr=log_file)
            else:
                result = subprocess.run(cmd, shell=True, check=True, text=True, cwd=cwd,
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE)
            # print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"Command failed: {cmd}")
            print(f"Error: {e.stderr}")
            assert False

    def intra_compress(self, in_img_path, out_hevc_path):
        cmd = f'ffmpeg -y -i {in_img_path} \
        -c:v libx265 -preset slow \
        -crf {self.qp} \
        -f hevc {out_hevc_path}'
        self.call_cmd(cmd, output_log=None)

    def hevc_to_png(self, hevc_path, png_path):
        cmd = f'ffmpeg -i {hevc_path} {png_path}'
        self.call_cmd(cmd)

    def H(self, image_tensor):
        image_tensor = (image_tensor + 1) / 2
        hm_imgs = []

        for i in range(image_tensor.size(0)):
            tmp_png = f"./tmp.png"
            out_hm = f"./hm_out.hevc"
            out_png = f"./hm_out.png"
            for file in [tmp_png, out_hm, out_png]:
                if os.path.exists(file): os.remove(file) 

            torchvision.utils.save_image(image_tensor[i], tmp_png)
            self.intra_compress(tmp_png, out_hm)
            self.hevc_to_png(out_hm, out_png)
            img = Image.open(out_png).convert("RGB")
            hm_img = self.transform(img).unsqueeze(0).to(image_tensor.device)
            hm_imgs.append(hm_img)

            for file in [tmp_png, out_hm, out_png]:
                if os.path.exists(file): os.remove(file)        
        
        return torch.cat(hm_imgs, dim=0)

    def H_pinv(self, x):
        return x


class HifiC():
    def __init__(self, mode, hific_root):
        if hific_root not in sys.path:
            sys.path.insert(0, hific_root)
        # from src.helpers import utils
        # from src.compression import compression_utils
        # from default_config import ModelModes
        self.mode = mode
        self.hific_root = hific_root
        
        # logging.disable(logging.CRITICAL)
        # with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
        self.model, _ = self.prepare_model(f'{hific_root}/weight/hific_{mode}.pt', f'{hific_root}/experiments')
        # logging.disable(logging.NOTSET)

    def make_deterministic(self, seed=3):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        np.random.seed(seed)

    def prepare_model(self, ckpt_path, input_dir):
        from src.helpers import utils
        from default_config import ModelModes
        self.make_deterministic()
        device = utils.get_device()
        logpath = os.path.join(input_dir, f'logs_tmp')
        logger = utils.logger_setup(logpath=logpath, filepath=os.path.abspath(__file__))
        loaded_args, model, _ = utils.load_model(ckpt_path, logger, device, model_mode=ModelModes.EVALUATION,
            current_args_d=None, prediction=True, strict=False, silent=True)

        # Build probability tables
        model.Hyperprior.hyperprior_entropy_model.build_tables()

        model.logger.disabled = True
        if os.path.exists(logpath): os.remove(logpath) 
        return model, loaded_args

    def H(self, image_tensor):
        from src.compression import compression_utils
        if self.mode == 'med': image_tensor = (image_tensor + 1.) / 2.
        imgs = []
        root = f'{self.hific_root}/experiments'
        
        logging.disable(logging.CRITICAL)
        with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
            with torch.no_grad():
                for i in range(image_tensor.size(0)):
                    compressed_output = self.model.compress(image_tensor[i].unsqueeze(0), silent=True)
                    actual_bpp, theoretical_bpp = compression_utils.save_compressed_format(compressed_output,
                        out_path=f'{root}/tmp.hfc')
                    # self.model.logger.info(f'Attained: {actual_bpp:.3f} bpp vs. theoretical: {theoretical_bpp:.3f} bpp.')

                    compressed_output = compression_utils.load_compressed_format(f'{root}/tmp.hfc')        
                    reconstruction = self.model.decompress(compressed_output)
                    imgs.append(reconstruction * 2 - 1)
        logging.disable(logging.NOTSET)

        return torch.cat(imgs, dim=0)
    
    def H_pinv(self, x):
        return x


class Quantization(H_functions):
    def __init__(self, n_bits):
        self.n_bits = n_bits

    def H(self, image):
        # Assert that image is in range [-1, 1]
        x = (image + 1) / 2
        x = torch.floor(x * (2 ** self.n_bits - 1) + 0.5) / (2 ** self.n_bits - 1)
        x = x * 2 - 1
        return x

    def H_pinv(self, x):
        return x


class HDR(H_functions):
    def __init__(self):
        pass

    def H(self, image):
        # Assert that image is in range [-1, 1]
        x = image
        x = torch.clip(x / 0.5, -1, 1)
        return x

    def H_pinv(self, x):
        return x * 0.5


class SRConv(H_functions):
    def mat_by_img(self, M, v, dim):
        return torch.matmul(M, v.reshape(v.shape[0] * self.channels, dim,
                        dim)).reshape(v.shape[0], self.channels, M.shape[0], dim)

    def img_by_mat(self, v, M, dim):
        return torch.matmul(v.reshape(v.shape[0] * self.channels, dim,
                        dim), M).reshape(v.shape[0], self.channels, dim, M.shape[1])

    def __init__(self, kernel, channels, img_dim, device, stride = 1):
        self.img_dim = img_dim
        self.channels = channels
        self.ratio = stride
        small_dim = img_dim // stride
        self.small_dim = small_dim
        #build 1D conv matrix
        H_small = torch.zeros(small_dim, img_dim, device=device)
        for i in range(stride//2, img_dim + stride//2, stride):
            for j in range(i - kernel.shape[0]//2, i + kernel.shape[0]//2):
                j_effective = j
                #reflective padding
                if j_effective < 0: j_effective = -j_effective-1
                if j_effective >= img_dim: j_effective = (img_dim - 1) - (j_effective - img_dim)
                #matrix building
                H_small[i // stride, j_effective] += kernel[j - i + kernel.shape[0]//2]
        #get the svd of the 1D conv
        self.U_small, self.singulars_small, self.V_small = torch.svd(H_small, some=False)
        ZERO = 3e-2
        self.singulars_small[self.singulars_small < ZERO] = 0
        #calculate the singular values of the big matrix
        self._singulars = torch.matmul(self.singulars_small.reshape(small_dim, 1), self.singulars_small.reshape(1, small_dim)).reshape(small_dim**2)
        #permutation for matching the singular values. See P_1 in Appendix D.5.
        self._perm = torch.Tensor([self.img_dim * i + j for i in range(self.small_dim) for j in range(self.small_dim)] + \
                                  [self.img_dim * i + j for i in range(self.small_dim) for j in range(self.small_dim, self.img_dim)]).to(device).long()

    def V(self, vec):
        #invert the permutation
        temp = torch.zeros(vec.shape[0], self.img_dim**2, self.channels, device=vec.device)
        temp[:, self._perm, :] = vec.clone().reshape(vec.shape[0], self.img_dim**2, self.channels)[:, :self._perm.shape[0], :]
        temp[:, self._perm.shape[0]:, :] = vec.clone().reshape(vec.shape[0], self.img_dim**2, self.channels)[:, self._perm.shape[0]:, :]
        temp = temp.permute(0, 2, 1)
        #multiply the image by V from the left and by V^T from the right
        out = self.mat_by_img(self.V_small, temp, self.img_dim)
        out = self.img_by_mat(out, self.V_small.transpose(0, 1), self.img_dim).reshape(vec.shape[0], -1)
        return out

    def Vt(self, vec):
        #multiply the image by V^T from the left and by V from the right
        temp = self.mat_by_img(self.V_small.transpose(0, 1), vec.clone(), self.img_dim)
        temp = self.img_by_mat(temp, self.V_small, self.img_dim).reshape(vec.shape[0], self.channels, -1)
        #permute the entries
        temp[:, :, :self._perm.shape[0]] = temp[:, :, self._perm]
        temp = temp.permute(0, 2, 1)
        return temp.reshape(vec.shape[0], -1)

    def U(self, vec):
        #invert the permutation
        temp = torch.zeros(vec.shape[0], self.small_dim**2, self.channels, device=vec.device)
        temp[:, :self.small_dim**2, :] = vec.clone().reshape(vec.shape[0], self.small_dim**2, self.channels)
        temp = temp.permute(0, 2, 1)
        #multiply the image by U from the left and by U^T from the right
        out = self.mat_by_img(self.U_small, temp, self.small_dim)
        out = self.img_by_mat(out, self.U_small.transpose(0, 1), self.small_dim).reshape(vec.shape[0], -1)
        return out

    def Ut(self, vec):
        #multiply the image by U^T from the left and by U from the right
        temp = self.mat_by_img(self.U_small.transpose(0, 1), vec.clone(), self.small_dim)
        temp = self.img_by_mat(temp, self.U_small, self.small_dim).reshape(vec.shape[0], self.channels, -1)
        #permute the entries
        temp = temp.permute(0, 2, 1)
        return temp.reshape(vec.shape[0], -1)

    def singulars(self):
        return self._singulars.repeat_interleave(3).reshape(-1)

    def add_zeros(self, vec):
        reshaped = vec.clone().reshape(vec.shape[0], -1)
        temp = torch.zeros((vec.shape[0], reshaped.shape[1] * self.ratio**2), device=vec.device)
        temp[:, :reshaped.shape[1]] = reshaped
        return temp



class Compose(H_functions):
    def __init__(self, Hs):
        self.Hs = Hs

    def set_indices(self, idx):
        # Assert that if there is inpainting, then it is at the last stage. 
        self.Hs[-1].set_indices(idx)

    def H(self, image):
        n, c, h, w = image.size()
        x = image
        # import ipdb; ipdb.set_trace()
        for i in range(len(self.Hs)):
            c, h = self.Hs[i].channels, self.Hs[i].img_dim
            x = x.reshape(-1, c, h, h)
            x = self.Hs[i].H(x)
        return x

    def H_pinv(self, image):
        x = image
        n = x.size(0)
        for i in reversed(range(len(self.Hs))):
            x = self.Hs[i].H_pinv(x)
            c, h = self.Hs[i].channels, self.Hs[i].img_dim
            x = x.reshape(-1, c, h, h)
        return x


def pdf(x, sigma=10):
    return torch.exp(torch.tensor([-0.5 * (x / sigma) ** 2]))


def save_mask(f, mask):
    m_npy = mask.astype(bool)
    m = np.packbits(m_npy, axis=None)
    shape = m_npy.shape
    np.savez(f, m=m, shape=shape)


def load_mask(f):
    d = np.load(f)
    m = d['m']
    shape = d['shape']
    m = np.unpackbits(m, count=np.prod(shape)).reshape(shape).view(bool)
    return m


def build_degredation_model(cfg: DictConfig):
    print(cfg.algo)
    deg =  getattr(cfg.algo, "deg", "deno")   #'inp_20ff'   #dfault value deno
    print('deg', deg)
    deg = deg.split('-')
    if len(deg) == 1:
        h, w, c = cfg.dataset.image_size, cfg.dataset.image_size, cfg.dataset.channels
        return build_one_degredation_model(cfg, h, w, c, deg[0])
    else:
        Hs = [] 
        h, w, c = cfg.dataset.image_size, cfg.dataset.image_size, cfg.dataset.channels
        for i in range(len(deg)):
            H = build_one_degredation_model(cfg, h, w, c, deg[i])
            Hs.append(H)
            
            if isinstance(H, SuperResolution) or isinstance(H, SRConv):
                h = h // H.ratio
                w = w // H.ratio
        
        return Compose(Hs)


def build_one_degredation_model(cfg, h, w, c, deg: str):
    device = torch.device("cuda")
    if deg == "deno":
        H = Denoising(c, w, device)
    elif deg[:3] == "inp":
        exp_root = cfg.exp.root
        deg_type = deg.split('_')[-1]
        masks_root = os.path.join(exp_root, 'masks', f'{deg_type}.npz')
        m = load_mask(masks_root)
        m = torch.from_numpy(m)
        H = Inpainting(c, w, m, device)
    elif deg[:3] == "in2":
        exp_root = cfg.exp.root
        deg_type = deg.split('_')[-1]
        masks_root = os.path.join(exp_root, 'masks', f'{deg_type}.npz')
        m = load_mask(masks_root)
        
        if h!=256 and w!=256:
            n = m.shape[0]
            m = np.reshape(m, (n, 3, 256, 256))
            m = np.repeat(m, w//256, axis=2)
            m = np.repeat(m, h//256, axis=3)
            m = np.reshape(m, (n, 3*h*w))
            # print('m.shape', m.shape)
            # print('m', m)
            # import pdb; pdb.set_trace()
            
        m = torch.from_numpy(m)
        H = Inpainting2(c, w, m, device)
    elif deg == "deblur_uni":
        H = Deblurring(torch.Tensor([1 / 9] * 9).to("cuda"), c, w, device)
    elif deg == "bdeblur_uni":
        H = BDeblur(torch.Tensor([1 / 9] * 9).to("cuda"), c, w, device)
    elif deg == "deblur_gauss":
        sigma = 25
        window = 13
        kernel = [pdf(t, sigma) for t in range(-(window-1)//2, (window-1)//2)]
        kernel = torch.Tensor([pdf(-2, sigma), pdf(-1, sigma), pdf(0, sigma), pdf(1, sigma), pdf(2, sigma)]).to(device)
        H = Deblurring(kernel / kernel.sum(), c, w, device)
    elif deg[:2] == "sr":
        downsample_by = int(deg[2:])
        H = SuperResolution(c, w, downsample_by, device)
    elif deg == "color":
        H = Colorization(w, device)
    elif 'jpeg' in deg:
        qf = int(deg[4:])
        H = JPEG(qf)
    elif 'mbt' in deg:
        qf = int(deg[3:])
        H = MBT_2018(qf)
    elif 'bmshj' in deg:
        qf = int(deg[5:])
        H = Hyperprior(qf)
    elif 'vvc' in deg:
        qp = int(deg[3:])
        H = VVC(qp, cfg.exp.vvc_root)
    elif 'hevc' in deg:
        qp = int(deg[4:])
        H = HEVC(qp)
    elif 'hific' in deg:
        hific_root = cfg.exp.hific_root
        mode = deg[5:]
        H = HifiC(mode, hific_root)
    elif 'quant' in deg:
        qf = int(deg[5:])
        H = Quantization(qf)
    elif deg == 'hdr':
        H = HDR()
    elif "bicubic" in deg:
        factor = int(deg[7:])
        def bicubic_kernel(x, a=-0.5):
            if abs(x) <= 1:
                return (a + 2)*abs(x)**3 - (a + 3)*abs(x)**2 + 1
            elif 1 < abs(x) and abs(x) < 2:
                return a*abs(x)**3 - 5*a*abs(x)**2 + 8*a*abs(x) - 4*a
            else:
                return 0
        k = np.zeros((factor * 4))
        for i in range(factor * 4):
            x = (1/factor)*(i- np.floor(factor*4/2) +0.5)
            k[i] = bicubic_kernel(x)
        k = k / np.sum(k)
        kernel = torch.from_numpy(k).float().to(device)
        H = SRConv(kernel / kernel.sum(), c, h, device, stride=factor)
    elif deg == "deblur_nl":
        
        # task_config = load_yaml(args.task_config)
        # measure_config = task_config['measurement']
        # H = NonlinearBlurOperator(device, **measure_config['operator'])
        #current_dir = os.getcwd()
        
        #current_dir = '/lustre/fsw/nvresearch/mmardani/source/latent-diffusion-sampling/pgdm'
        current_dir = '/home/mmardani/research/stable-diffusion-sampling-gitlab/pgdm'
        
        opt_yml_path = os.path.join(current_dir, 'bkse/options/generate_blur/default.yml')
        #opt_yml_path = os.path.join(current_dir, 'bkse/options/generic_deblur/default.yml')
        print('opt_yml_path', opt_yml_path)
        H = NonlinearBlurOperator(device=device, opt_yml_path=opt_yml_path, current_dir=current_dir)
    elif deg == "phase_retrieval":
        oversample=2.0
        H = PhaseRetrievalOperator(oversample=oversample, device=device)
    else:
        raise ValueError(f"Degredation model {deg} does not exist.")

    return H


def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def get_degreadation_image(y_0, H, cfg):
    c, w = cfg.dataset.channels, cfg.dataset.image_size
    pinv_y = H.H_pinv(y_0).reshape(-1, c, w, w)
    return pinv_y





#nonlinear operators

from abc import ABC, abstractmethod

class NonLinearOperator(ABC):
    @abstractmethod
    def forward(self, data, **kwargs):
        pass

    def project(self, data, measurement, **kwargs):
        return data + measurement - self.forward(data) 

#@register_operator(name='nonlinear_blur')
class NonlinearBlurOperator(NonLinearOperator, H_functions):
    def __init__(self, opt_yml_path, current_dir, device):
        self.device = device
        self.blur_model = self.prepare_nonlinear_blur_model(opt_yml_path, current_dir)     
         
    def prepare_nonlinear_blur_model(self, opt_yml_path, current_dir):
        '''
        Nonlinear deblur requires external codes (bkse).
        '''
        from bkse.models.kernel_encoding.kernel_wizard import KernelWizard

        with open(opt_yml_path, "r") as f:
            opt = yaml.safe_load(f)
            opt = opt['KernelWizard']
            model_path = opt["pretrained"]   #this nees to be changed to the original one
            #model_path = os.path.join(os.dirname(os.dirname(opt_yml_path)), model_path)
            #current_dir = os.getcwd()
            #current_dir = '/lustre/fsw/nvresearch/mmardani/source/latent-diffusion-sampling/pgdm'
            #current_dir = '/home/mmardani/research/stable-diffusion-sampling-gitlab/pgdm'
            model_path = os.path.join(current_dir, model_path)
            print('model_path', model_path)
        blur_model = KernelWizard(opt)
        blur_model.eval()
        
        # Disable gradient computation for all parameters
        with torch.no_grad():
            for param in blur_model.parameters():
                param.requires_grad_(False)
                
        # Load the model state_dict directly
        state_dict = torch.load(model_path, map_location=self.device)
        state_dict = {'state_dict': state_dict}

        blur_model.load_state_dict(state_dict, strict=False)
        blur_model = blur_model.to(self.device)
        
        return blur_model
    
    def forward(self, data, **kwargs):
        random_kernel = torch.randn(data.shape[0], 512, 2, 2).to(self.device) * 1.2
        data = (data + 1.0) / 2.0  #[-1, 1] -> [0, 1]
        blurred = self.blur_model.adaptKernel(data, kernel=random_kernel)
        blurred = (blurred * 2.0 - 1.0).clamp(-1, 1) #[0, 1] -> [-1, 1]
        return blurred
    
    def H(self, data, **kwargs):
        return self.forward(data, **kwargs)
    
    def H_pinv(self, x):
        return self.H(x)
    
    

# #@register_operator(name='phase_retrieval')
# class PhaseRetrievalOperator(NonLinearOperator):
#     def __init__(self, oversample, device):
#         self.pad = int((oversample / 8.0) * 256)
#         self.device = device
        
#     def forward(self, data, **kwargs):
#         padded = F.pad(data, (self.pad, self.pad, self.pad, self.pad))
#         amplitude = fft2_m(padded).abs()
#         return amplitude
    
#     def H(self, data, **kwargs):
#         return self.forward(data, **kwargs)
    
#     def H_pinv(self, x):
#         x = ifft2_m(x).abs()
#         x = self.undo_padding(x, self.pad, self.pad, self.pad, self.pad)
#         return x
    
#     def undo_padding(self, tensor, pad_left, pad_right, pad_top, pad_bottom):
#         # Assuming 'tensor' is the 4D tensor
#         # 'pad_left', 'pad_right', 'pad_top', 'pad_bottom' are the padding values
#         if tensor.dim() != 4:
#             raise ValueError("Input tensor should have 4 dimensions.")
#         return tensor[:, :, pad_top : -pad_bottom, pad_left : -pad_right]