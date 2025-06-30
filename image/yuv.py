import numpy as np
import os
import re
from PIL import Image

YCBCR_WEIGHTS = {
    # Spec: (K_r, K_g, K_b) with K_g = 1 - K_r - K_b
    "ITU-R_BT.709": (0.2126, 0.7152, 0.0722)
}

FILTER_TRANSLATION={
    "bicubic":Image.BICUBIC,
    "bilinear":Image.BILINEAR,
    "nearest":Image.NEAREST,
    "lanczos":Image.LANCZOS,
}

def yuv_identify_param(filename):

    m = re.search(r'_([0-9]+)x([0-9]+)_', filename)
    if not m:
        print("yuv name mis-formatted: we need the resolution _wxh_ inside the name")
        return 0,0,0

    w=int(m.group(1))
    h=int(m.group(2))

    bit_depth=8
    m = re.search(r'_10b', filename)
    if m:
        bit_depth=10

    colorspace='bt709'
    m = re.search(r'ycocg', filename)
    if m:
        colorspace='ycocg'

    format='444'
    m = re.search(r'420p', filename)
    if m:
        format='420'

    return w,h,format,bit_depth,colorspace

def read_video(filename: str, frame_idx: int, w:int, h:int, bit_depth:int=8, format:str="444"):
    """From a filename /a/b/c.yuv, read the desired frame_index
    and return a dictionary of tensor containing the YUV values:
        {
            'Y': [1, 1, H, W],
            'U': [1, 1, H / x, W / x],
            'V': [1, 1, H / x, W / x],
        }
    The YUV values are return in [0., 255.] for y and [-128,+127] for the chroma

    /!\ bit depth and resolution are inferred from the filename which should
        be something like:
            B-MarketPlace_1920x1080_60p_yuv420_10b.yuv


    Args:
        filename (str): Absolute path of the video to load
        frame_idx (int): Index of the frame to load, starting at 0.

    Returns:
        DictTensorYUV: The YUV values (see format above).
    """

    if "420" in format:
        w_uv, h_uv = [x // 2 for x in [w, h]]
    else:
        w_uv, h_uv = [ w, h ]

    # Switch between 8 bit file and 10 bit
    byte_per_value = 1 if bit_depth == 8 else 2

    # We only handle YUV420 for now
    n_val_y = h * w
    n_val_uv = h_uv * w_uv
    n_val_per_frame = n_val_y + 2 * n_val_uv

    n_bytes_y = n_val_y * byte_per_value
    n_bytes_uv = n_val_uv * byte_per_value
    n_bytes_per_frame = n_bytes_y + 2 * n_bytes_uv

    # Read the required frame and put it in a 1d tensor
    raw_video = np.array(
        np.memmap(
            filename,
            mode='r',
            shape=(n_val_per_frame),
            offset=n_bytes_per_frame * frame_idx,
            dtype=np.uint16 if bit_depth == 10 else np.uint8
        )
    )

    # Read the different values from raw video and store them inside y, u and v
    ptr = 0
    y = raw_video[ptr: ptr + n_val_y ].reshape(h   ,w    ).astype(np.float32)
    ptr += n_val_y
    u = raw_video[ptr: ptr + n_val_uv].reshape(h_uv, w_uv).astype(np.float32)
    ptr += n_val_uv
    v = raw_video[ptr: ptr + n_val_uv].reshape(h_uv, w_uv).astype(np.float32)

    if bit_depth==10:
        norm_factor = 4.
        y/=norm_factor
        u/=norm_factor
        v/=norm_factor

    cb=u-128.
    cr=v-128.

    return [y,cb,cr]

# y  =      0 to 255
# cb =   -128 to 127
# cr =   -128 to 127

def write_yuv(y,cr,cb, filename: str, bitdepth: int = 8):

    h,w = y.shape
    h2,w2 = cb.shape

    cr = cr+128
    cb = cb+128

    y = y.reshape((h*w,1))
    cr=cr.reshape((h2*w2,1))
    cb=cb.reshape((h2*w2,1))

    dtype = np.uint8
    max_clip = 2**bitdepth -1
    if bitdepth == 10:
        dtype = np.uint16
        y  = 4 * y
        cr = 4 * cr
        cb = 4 * cb

    raw_data = np.round(np.concatenate([y, cr, cb])).astype(dtype)
    raw_data = np.clip(raw_data,0,max_clip)

    np.memmap.tofile(raw_data, filename)

def ycbcr2rgb(y, cb, cr):
    """YCbCr to RGB conversion f
    Using ITU-R BT.709 coefficients.

    Args:
        y,cb,cr :
    Returns:
        rgb : converted tensor
    """
    Kr, Kg, Kb = YCBCR_WEIGHTS["ITU-R_BT.709"]
    r = y + (2 - 2 * Kr) * (cr - 0.5)
    b = y + (2 - 2 * Kb) * (cb - 0.5)
    g = (y - Kr * r - Kb * b) / Kg
    return [r,g,b]

def rgb2ycbcr(r,g,b):
    """RGB to YCbCr conversion for torch Tensor.
    Using ITU-R BT.709 coefficients.

    Args:
        r,g,b 2d planes

    Returns:
        ycbcr (torch.Tensor): converted tensor
    """
    Kr, Kg, Kb = YCBCR_WEIGHTS["ITU-R_BT.709"]
    y = Kr * r + Kg * g + Kb * b
    cb = 0.5 * (b - y) / (1 - Kb) + 0.5
    cr = 0.5 * (r - y) / (1 - Kr) + 0.5
    return [y,cb,cr]

def YCoCg2rgb(y, Co, Cg):
    """YCoCg to RGB conversion f

    Args:
        y,cb,cr :
    Returns:
        rgb : converted
    """
    g   = y   + Cg
    tmp = y   - Cg
    r   = tmp + Co
    b   = tmp - Co

    return [r,g,b]

def rgb2YCoCg(r,g,b):
    """RGB to YCoCg conversion for torch Tensor.

    Args:r,g,b

    Returns:
        YCoCg : converted
    """

    y  = .25 * r + .50 * g + .25 * b
    Co = .50 * r           - .50 * b
    Cg =-.25 * r + .50 * g - .25 * b

    return [y,Co,Cg]

def yuv444420(u,v,filter:str='bicubic'):

    # https://stackoverflow.com/questions/35381551/fast-interpolation-resample-of-numpy-array-python
    im = Image.fromarray(u)
    ud = im.resize((im.width//2, im.height//2),resample=FILTER_TRANSLATION[filter])
    ud=np.array(ud,dtype=float)

    im = Image.fromarray(v)
    vd = im.resize((im.width//2, im.height//2),resample=FILTER_TRANSLATION[filter])
    vd=np.array(vd,dtype=float)

    return [ud,vd]

def yuv420_444(u,v,filter:str='bicubic'):

    # https://stackoverflow.com/questions/35381551/fast-interpolation-resample-of-numpy-array-python
    im = Image.fromarray(u)
    ud = im.resize((im.width*2, im.height*2),resample=FILTER_TRANSLATION[filter])
    ud=np.array(ud,dtype=float)

    im = Image.fromarray(v)
    vd = im.resize((im.width*2, im.height*2),resample=FILTER_TRANSLATION[filter])
    vd=np.array(vd,dtype=float)

    return [ud,vd]
