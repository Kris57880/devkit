# CLIC devkit

This contains code for example submissions to the CLIC challenge. It is based on the VVC
baselines provided for the challenge.

The `image/` directory contains code that decodes into `.png` files as needed for the
image compression tracks. The `video/` directory has code decoding into raw `.yuv` files,
as needed for the video compression tracks. Note that VVC operates only on `.yuv` files,
so the image submission also contains a script that converts to `.png` after decoding.

The main entry point in both cases is the `decode` script. Both scripts expect the
compressed data (bit streams) in a zip file named `bs.zip` (not provided here).

## Docker images

The following Docker images are provided for running the submitted scripts. They are based on PyTorch:

```
800102672637.dkr.ecr.us-east-1.amazonaws.com/compression (CPU)
800102672637.dkr.ecr.us-east-1.amazonaws.com/compression:gpu (GPU)
```

If you need significant changes to the environment your decoder runs in, please email the [CLIC mailing list](clic-list@googlegroups.com) and request a custom image.
