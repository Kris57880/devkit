#!/usr/bin/env python3

from PIL import Image
import sys
import numpy as np
import argparse

from typing import Tuple, Union
from yuv import read_video,write_yuv,ycbcr2rgb,yuv_identify_param,yuv420_444,YCoCg2rgb


parser = argparse.ArgumentParser()
parser.add_argument('filename')
parser.add_argument('-o', help='name of the output file', type=str, default="rec.png")
parser.add_argument('-s', help='y plane size wxh', type=str, default=None)
parser.add_argument('-v', help='verbosity level', type=int, default=0)
parser.add_argument('-bit_depth', help='yuv bitdepth', choices=[8,10], type=int, default=10)
parser.add_argument('-format', help='yuv sampling', choices=['420', '444'], default="444")
parser.add_argument('-filter', help='filter for chroma downsampling', choices=['bicubic', 'lanczos','bilinear','nearest'], default="lanczos")
parser.add_argument('-colorspace', choices=['bt709', 'ycocg'], default="bt709")


args = parser.parse_args()


filename=args.filename
outname=args.o
bit_depth=args.bit_depth
format=args.format
colorspace=args.colorspace

if args.s:
    w,h=args.s.split("x")
    w=int(w)
    h=int(h)
else:
    w,h,format,bit_depth,colorspace = yuv_identify_param(filename)
    if w==0 or h==0:
        print("unable to identify the resolution ")
        exit(1)

if args.v >0:
    print(w,h,bit_depth)

y,cb,cr = read_video(filename, 0, w, h, bit_depth=bit_depth, format=format)

if args.v >0:
    print(f"y :{np.min( y):7.1f}-{np.max( y):7.1f}")
    print(f"cb:{np.min(cb):7.1f}-{np.max(cb):7.1f}")
    print(f"cr:{np.min(cr):7.1f}-{np.max(cr):7.1f}")


if "420" in format:
    cb,cr = yuv420_444(cb,cr,filter=args.filter)

if "bt709" in colorspace:
    r,g,b = ycbcr2rgb(y,cb,cr)
else:
    r,g,b = YCoCg2rgb(y,cb,cr)

if args.v >0:
    print(f"r :{np.min( r):7.1f}-{np.max( r):7.1f}")
    print(f"g :{np.min( g):7.1f}-{np.max( g):7.1f}")
    print(f"b :{np.min( b):7.1f}-{np.max( b):7.1f}")


rgb=np.stack((r,g,b),axis=2)
rgb=np.round(rgb)

img_numpy = np.array(np.clip(rgb,0,255), dtype=np.uint8)

img = Image.fromarray(img_numpy, "RGB")
img.save(outname)

