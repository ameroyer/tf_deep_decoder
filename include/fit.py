import os
from datetime import datetime

import numpy as np
import tensorflow as tf

def get_num_params():
    total_parameters = 0
    for variable in tf.trainable_variables(scope="DeepDecoder"):
        params = 1
        for dim in variable.get_shape():
            params *= dim.value
        total_parameters += params
    return total_parameters

def fit(net,
        img_noisy,
        num_channels,
        img_clean,
        num_iter=5000,
        LR=0.01,
        OPTIMIZER='adam',
        opt_input=False,
        reg_noise_std=0,
        reg_noise_decayevery=100000,
        reg_noise_decay_rate=0.7,
        mask=None,
        apply_f=None,
        lr_decay_epoch=0,
        lr_decay_rate=0.65,
        net_input=None,
        find_best=False,
        weight_decay=0,
        device='gpu',
        verbose=False
       ):
    """Fit a model.
    
        Args: 
        net: one of the network functions defined in `net.py`
        img_noisy: Noisy observation y used to "train" the network
        num_channels: Number of upsample channels
        img_clean: Clean observation x, used only for informative MSE (not used to learn the weights)
    """
    
    with tf.Graph().as_default():
        # Global step
        global_step = tf.train.get_or_create_global_step()
            
        with tf.device('/%s' % device):        
            # Define inputs
            if net_input is not None:
                if opt_input:
                    net_input = tf.Variable(opt_input, trainable=True, name='net_input')
                else:
                    net_input = tf.stop_gradient(net_input)
            else:
                # feed uniform noise into the network 
                totalupsample = 2**len(num_channels)
                width = int(img_clean.data.shape[1] / totalupsample)
                height = int(img_clean.data.shape[2] / totalupsample)
                shape = [1, width, height, num_channels[0]]
                print("shape: ", shape)
                if opt_input:
                    net_input = tf.Variable(tf.random_uniform(shape) * 1. / 10, trainable=True, name='net_input')
                else:
                    net_input = tf.constant(np.random.uniform(size=shape).astype(np.float32) * 1. / 10, name='net_input')
            net_input_saved = net_input

            # Add some random noise to inputs
            if reg_noise_std > 0:
                reg_noise = tf.train.exponential_decay(reg_noise_std, global_step, reg_noise_decayevery,
                                                       reg_noise_decay_rate, staircase=True)
                net_input += tf.random_uniform(tf.shape(net_input)) * reg_noise
            
            # Feed-Forward 
            feed_forward = tf.make_template("DeepDecoder", net)
            net_output = feed_forward(net_input)       
            net_output_from_saved = feed_forward(net_input_saved)
        
            # Training Loss
            mse = tf.losses.mean_squared_error
            if mask is not None:
                loss = mse(net_output * mask, img_noisy * mask)
            elif apply_f:
                loss = mse(apply_f(net_output), img_noisy)
            else:
                loss = mse(net_output, img_noisy)

            # Train operation
            if lr_decay_epoch > 0:
                LR = tf.train.exponential_decay(LR, global_step, lr_decay_epoch, lr_decay_rate, staircase=True)

            # TODO: weight decay
            if OPTIMIZER == 'SGD':
                print("optimize with SGD", LR)
                optimizer = torch.optim.GradientDescentOptimizer(LR, 0.9)
            elif OPTIMIZER == 'adam':
                print("optimize with adam", LR)
                optimizer = tf.train.AdamOptimizer(LR)
            elif OPTIMIZER == 'LBFGS':
                raise NotImplementedError('LBFGS Optimizer')

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = optimizer.minimize(loss, global_step=global_step)    

            # Additional output information
            noise_energy = mse(img_noisy, img_clean)
            true_loss = mse(net_output, img_clean)  
            if reg_noise_std > 0: 
                true_loss_original = mse(net_output_from_saved, img_clean)

        with tf.Session() as sess:
            # Init            
            mse_wrt_noisy = [0.] * num_iter
            mse_wrt_truth = [0.] * num_iter
            sess.run(tf.global_variables_initializer())            
            if find_best:
                save_log_dir = os.path.join('log', datetime.now().strftime("%m-%d_%H-%M"))
                if verbose:
                    print('Save net in', save_log_dir)
                saver = tf.train.Saver(max_to_keep=1)
                saver.save(sess, os.path.join(save_log_dir, 'net'), global_step=0)
                best_mse = 1000000.0
                best_true_mse = 0.
                best_img = sess.run(net_output_from_saved)
            
            # Optimize
            num_params = get_num_params()
            noise_energy_ = sess.run(noise_energy)
            sess.graph.finalize()
            print('\x1b[37mFinal graph size: %.2f MB\x1b[0m' % (
                tf.get_default_graph().as_graph_def().ByteSize() / 10e6))

            for i in range(num_iter):
                if reg_noise_std <= 0:  
                    loss_, true_loss_, _ = sess.run([loss, true_loss, train_op])
                    true_loss_original_ = true_loss_
                else:
                    loss_, true_loss_, true_loss_original_, _ = sess.run([
                        loss, true_loss, true_loss_original, train_op])
                mse_wrt_noisy[i] = loss_
                mse_wrt_truth[i] = true_loss_
                        
                # Display
                if i > 0 and i % 10 == 0:
                    print ('\r[Iteration %05d] loss=%.5f  true loss=%.5f  true loss orig=%.5f  noise energy=%.5f' % (
                        i, loss_, true_loss_, true_loss_original_, noise_energy_), end='')  
                
                # Best net
                if find_best and best_mse > 1.005 * loss_:
                    best_mse = loss_
                    best_true_mse = true_loss_
                    best_img = sess.run(net_output_from_saved)
                    saver.save(sess, os.path.join(save_log_dir, 'net'), global_step=i + 1)                  
                        
            # Return final image or best found so far if `find_best`
            if find_best:
                out_img = best_img
                print()
                print('Best MSE (wrt noisy)', best_mse)
                print('MSE (wrt true)', best_true_mse)
            else:
                out_img = sess.run(net_output_from_saved)
            if verbose:
                return mse_wrt_noisy, mse_wrt_truth, sess.run(net_input_saved), out_img, num_params
            else:
                return mse_wrt_noisy, mse_wrt_truth, sess.run(net_input_saved), out_img