import tensorflow as tf

def _pad(x, kernel_size, pad):
    to_pad = int((kernel_size - 1) / 2)
    if to_pad > 0 and pad == 'reflection':
        return tf.pad(x, ((0, 0), (to_pad, to_pad), (to_pad, to_pad), (0, 0)), mode='REFLECT')
    else:
        return x

def _upsample(x, upsample_mode):
    if upsample_mode == 'none':
        return x
    
    else:
        w = tf.shape(x)[1]
        h = tf.shape(x)[2]
        new_shape = tf.stack([w * 2, h * 2], axis=0)
        try:
            # align_corners = True necessary for bilinear interpolation
            x = tf.image.resize_images(x, new_shape, align_corners=True,
                                       method=getattr(tf.image.ResizeMethod, upsample_mode.upper()))
            return x
        except AttributeError:
            raise NotImplementedError('%s rescaling' % upsample_mode)
            
def _bn(x, bn_affine):
    return tf.layers.batch_normalization(x, trainable=bn_affine, momentum=0.9, epsilon=1e-5, training=True)      


def decodernw(inputs,
              num_output_channels=3, 
              num_channels_up=[128] * 5,
              filter_size_up=1,
              need_sigmoid=True, 
              pad='reflection',
              upsample_mode='bilinear', 
              act_fun=tf.nn.relu, # tf.nn.leaky_relu 
              bn_before_act=False,
              bn_affine=True,
              upsample_first=True
             ):
    """Deep Decoder.
       Takes as inputs a 4D Tensor (batch, width, height, channels)"""
    ## Configure
    num_channels_up = num_channels_up[1:] + [num_channels_up[-1], num_channels_up[-1]]
    n_scales = len(num_channels_up) 
    
    if not (isinstance(filter_size_up, list) or isinstance(filter_size_up, tuple)) :
        filter_size_up = [filter_size_up] * n_scales                                                  
        
    ## Deep Decoder
    net = inputs
    for i, (num_channels, kernel_size) in enumerate(zip(num_channels_up, filter_size_up)):       
        # Upsample (first)
        if upsample_first and i != 0:
            net = _upsample(net, upsample_mode)

        # Conv        
        net = _pad(net, kernel_size, pad)
        net = tf.layers.conv2d(net, num_channels, kernel_size=kernel_size, strides=1,
                               activation=None, padding='valid', use_bias=False)

        # Upsample (second)
        if not upsample_first and i != len(num_channels_up) - 1:
            net = _upsample(net, upsample_mode)

        # Batch Norm + activation
        if bn_before_act: 
            net = _bn(net, bn_affine)           
        net = act_fun(net)
        if not bn_before_act: 
            net = _bn(net, bn_affine) 
                
    # Final convolution
    kernel_size = 1
    net = _pad(net, kernel_size, pad)
    net = tf.layers.conv2d(net, num_output_channels, kernel_size, strides=1, 
                           activation=None, padding='valid', use_bias=False)
    if need_sigmoid:
        net = tf.nn.sigmoid(net)    
    return net