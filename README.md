# Deep Decoder
---
The code is based on the official `Pytorch` implementation of Deep Decoders [available here](https://github.com/reinhardh/supplement_deep_decoder).

The repository contains a `Tensorflow` implementation of **Deep Decoders**, as described in the paper *Deep Decoder: Concise Image Representations from Untrained Non-convolutional Networks*, Reinhard Heckel and Paul Hand, ICLR 2019. 

#### Notes 
  * The code has been tested with `Tensorflow 1.12` and `Python 3.5+`. Other dependecies include `numpy`, `matplotlib` and `skimage` (for reading/saving images only). 
  * The `upsample_first` argument in the decoder is inverted, to better match its meaning (i.e., when `True`, the upsampling operation occurs before the linear combination of channels)
  * Currently the PSNR values are often slightly lower than the original implementation  (possibly due to a different behavior in `tf.image.resize_bilinear` or some other error)
  * `LBFGS` optimizer and `weight decay` are not implemented (but also not used in any of the present notebooks)