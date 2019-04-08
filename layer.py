import tensorflow as tf
import keras
from keras import backend as K
from utils import _conv2d_wrapper, _get_weights_wrapper

def softmax(x, axis=-1):
    ex = K.exp(x - K.max(x, axis=axis, keepdims=True))
    return ex/K.sum(ex, axis=axis, keepdims=True)

def squash_v1(x, axis=-1):
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True) + K.epsilon()
    scale = K.sqrt(s_squared_norm)/ (0.5 + s_squared_norm)
    return scale * x

def squash_v0(s, axis=-1, epsilon=1e-7, name=None):
    s_squared_norm = K.sum(K.square(s), axis, keepdims=True) + K.epsilon()
    safe_norm = K.sqrt(s_squared_norm)
    scale = 1 - tf.exp(-safe_norm)
    return scale * s / safe_norm
   
def routing(u_hat_vecs, beta_a, iterations, output_capsule_num, i_activations): #[2425,16,48,16]
    b = keras.backend.zeros_like(u_hat_vecs[:,:,:,0])  #(2425, 16, 48)
    if i_activations is not None:
        i_activations = i_activations[...,tf.newaxis]
    for i in range(iterations):
        if False:
            leak = tf.zeros_like(b, optimize=True)
            leak = tf.reduce_sum(leak, axis=1, keep_dims=True)
            leaky_logits = tf.concat([leak, b], axis=1)
            leaky_routing = tf.nn.softmax(leaky_logits, dim=1)        
            c = tf.split(leaky_routing, [1, output_capsule_num], axis=1)[1]
        else:
            c = softmax(b, 1)   
#        if i_activations is not None:
#            tf.transpose(tf.transpose(c, perm=[0,2,1]) * i_activations, perm=[0,2,1]) 
        outputs = squash_v1(K.batch_dot(c, u_hat_vecs, [2, 2]))#(2425, 16, 48) #[2425,16,48,16]   =>[2425,16,16]  相当于tf.squeeze(tf.matmul(tf.expand_dims(c,2),u_hat_vecs))
        if i < iterations - 1:
            b = b + K.batch_dot(outputs, u_hat_vecs, [2, 3])   #(2425, 16, 16) #[2425,16,48,16]   =>[2425,16,48]  相当于tf.squeeze(tf.matmul(tf.expand_dims(outputs,2),u_hat_vecs,transpose_b=True))
    poses = outputs #[2425,16,16]
    activations = K.sqrt(K.sum(K.square(poses), 2))#[2425,16]
    return poses, activations


def vec_transformationByConv(poses, input_capsule_dim, input_capsule_num, output_capsule_dim, output_capsule_num):    # #(2425, 48, 16)  16 48 16 16
    kernel = _get_weights_wrapper(
      name='weights', shape=[1, input_capsule_dim, output_capsule_dim*output_capsule_num], weights_decay_factor=0.0
    )
    tf.logging.info('poses: {}'.format(poses.get_shape()))   
    tf.logging.info('kernel: {}'.format(kernel.get_shape()))
    u_hat_vecs = keras.backend.conv1d(poses, kernel)  # $\hat u$   # #(2425, 48,1,16*16)
    u_hat_vecs = keras.backend.reshape(u_hat_vecs, (-1, input_capsule_num, output_capsule_num, output_capsule_dim))  #[2425,48,16,16]
    u_hat_vecs = keras.backend.permute_dimensions(u_hat_vecs, (0, 2, 1, 3))#[2425,16,48,16]
    return u_hat_vecs  #[2425,16,48,16]

def vec_transformationByMat(poses, input_capsule_dim, input_capsule_num, output_capsule_dim, output_capsule_num, shared=True):                        
    inputs_poses_shape = poses.get_shape().as_list()
    poses = poses[..., tf.newaxis, :]        
    poses = tf.tile(
              poses, [1, 1, output_capsule_num, 1]
            )    
    if shared:
        kernel = _get_weights_wrapper(
          name='weights', shape=[1, 1, output_capsule_num, output_capsule_dim, input_capsule_dim], weights_decay_factor=0.0
        )
        kernel = tf.tile(
                  kernel, [inputs_poses_shape[0], input_capsule_num, 1, 1, 1]
                )
    else:
        kernel = _get_weights_wrapper(
          name='weights', shape=[1, input_capsule_num, output_capsule_num, output_capsule_dim, input_capsule_dim], weights_decay_factor=0.0
        )
        kernel = tf.tile(
                  kernel, [inputs_poses_shape[0], 1, 1, 1, 1]
                )
    tf.logging.info('poses: {}'.format(poses[...,tf.newaxis].get_shape()))   
    tf.logging.info('kernel: {}'.format(kernel.get_shape()))
    u_hat_vecs = tf.squeeze(tf.matmul(kernel, poses[...,tf.newaxis]),axis=-1)
    u_hat_vecs = keras.backend.permute_dimensions(u_hat_vecs, (0, 2, 1, 3))
    return u_hat_vecs

def capsules_init(inputs, shape, strides, padding, pose_shape, add_bias, name): #nets, shape=[1, 1, 32, 16], strides=[1, 1, 1, 1],
                                                                                # padding='VALID', pose_shape=16, add_bias=True, name='primary')
                                                                                # nets = capsule_conv_layer(nets, shape=[3, 1, 16, 16], strides=[1, 1, 1, 1],
                                                                                # iterations=3, name='conv2'
    with tf.variable_scope(name):   
        poses = _conv2d_wrapper(
          inputs,
          shape=shape[0:-1] + [shape[-1] * pose_shape],  # [1, 1, 32, 16*16]
          strides=strides,
          padding=padding,
          add_bias=add_bias,
          activation_fn=None,
          name='pose_stacked'
        )        
        poses_shape = poses.get_shape().as_list()   #[batch , len-2, 1, 16*16]
        poses = tf.reshape(
                    poses, [
                        -1, poses_shape[1], poses_shape[2], shape[-1], pose_shape
                    ])         # [batch , len-2 , 1 , 16, 16]
        beta_a = _get_weights_wrapper(
                        name='beta_a', shape=[1, shape[-1]]
                    )    
        poses = squash_v1(poses, axis=-1)   # [batch , len-2 , 1 , 16, 16]
        activations = K.sqrt(K.sum(K.square(poses), axis=-1)) + beta_a         # [batch , len-2 , 1 , 16]
        tf.logging.info("prim poses dimension:{}".format(poses.get_shape()))

    return poses, activations  # [batch , len-2 , 1 , 16, 16]  # [batch , len-2 , 1 , 16]

def capsule_fc_layer(nets, output_capsule_num, iterations, name):
    with tf.variable_scope(name):   
        poses, i_activations = nets      #25, 97*16, 16   #25, 97*16
        input_pose_shape = poses.get_shape().as_list()

        u_hat_vecs = vec_transformationByConv(
                      poses,
                      input_pose_shape[-1], input_pose_shape[1],
                      input_pose_shape[-1], output_capsule_num,
                      )#25,output_capsule_num, 97*16, 16
        
        tf.logging.info('votes shape: {}'.format(u_hat_vecs.get_shape()))
        
        beta_a = _get_weights_wrapper(
                name='beta_a', shape=[1, output_capsule_num]
                )

        poses, activations = routing(u_hat_vecs, beta_a, iterations, output_capsule_num, i_activations) ##25,output_capsule_num , 16
        
        tf.logging.info('capsule fc shape: {}'.format(poses.get_shape()))   
        
    return poses, activations

def capsule_flatten(nets):
    poses, activations = nets  #25, 97, 1, 16, 16   #25, 97, 1, 16
    input_pose_shape = poses.get_shape().as_list()
    
    poses = tf.reshape(poses, [
                    -1, input_pose_shape[1]*input_pose_shape[2]*input_pose_shape[3], input_pose_shape[-1]]) 
    activations = tf.reshape(activations, [
                    -1, input_pose_shape[1]*input_pose_shape[2]*input_pose_shape[3]])
    tf.logging.info("flatten poses dimension:{}".format(poses.get_shape()))
    tf.logging.info("flatten activations dimension:{}".format(activations.get_shape()))

    return poses, activations

def capsule_conv_layer(nets, shape, strides, iterations, name):   #nets, shape=[3, 1, 16, 16], strides=[1, 1, 1, 1], iterations=3
    with tf.variable_scope(name):              
        poses, i_activations = nets  #(25, 99, 1, 16, 16)  i_activations:(25, 99, 1, 16)
        
        inputs_poses_shape = poses.get_shape().as_list()

        hk_offsets = [
          [(h_offset + k_offset) for k_offset in range(0, shape[0])] for h_offset in
          range(0, inputs_poses_shape[1] + 1 - shape[0], strides[1])
        ] #
        wk_offsets = [
          [(w_offset + k_offset) for k_offset in range(0, shape[1])] for w_offset in
          range(0, inputs_poses_shape[2] + 1 - shape[1], strides[2])
        ]
        tmpa = tf.gather(poses, hk_offsets, axis=1, name='gather_poses_height_kernel'),  # [25, 97,3, 1, 16, 16]
        tmpb = tf.gather(tmpa, wk_offsets, axis=3, name='gather_poses_width_kernel')  # [25, 97,3,1,1, 16, 16]
        tf.logging.info('tmpa shape: {}\ntmpb:{}'.format(tmpa.get_shape(),tmpb.get_shape()))
        inputs_poses_patches = tf.transpose(tmpb,perm=[0, 1, 3, 2, 4, 5, 6], name='inputs_poses_patches')# [25, 97,1,3,1, 16, 16]  窗口个数97 窗口大小3
        tf.logging.info('i_poses_patches shape: {}'.format(inputs_poses_patches.get_shape()))
    
        inputs_poses_shape = inputs_poses_patches.get_shape().as_list()# [25, 97,1,3,1, 16, 16]
        inputs_poses_patches = tf.reshape(inputs_poses_patches, [
                                -1, shape[0]*shape[1]*shape[2], inputs_poses_shape[-1]
                                ])  #(2425, 48, 16)

        i_activations_patches = tf.transpose(
          tf.gather(
            tf.gather(
              i_activations, hk_offsets, axis=1, name='gather_activations_height_kernel'
            ), wk_offsets, axis=3, name='gather_activations_width_kernel'
          ), perm=[0, 1, 3, 2, 4, 5], name='inputs_activations_patches'
        ) #(25, 97, 1, 3, 1, 16)
        tf.logging.info('i_activations_patches shape: {}'.format(i_activations_patches.get_shape()))
        i_activations_patches = tf.reshape(i_activations_patches, [
                                -1, shape[0]*shape[1]*shape[2]]
                                )#(2425, 48)
        u_hat_vecs = vec_transformationByConv(   ### W 映射矩阵
                  inputs_poses_patches,   #(2425, 48, 16)
                  inputs_poses_shape[-1], shape[0]*shape[1]*shape[2],  #16 48
                  inputs_poses_shape[-1], shape[3],# 16 16
                  ) #[2425,16,48,16]
        tf.logging.info('capsule conv votes shape: {}'.format(u_hat_vecs.get_shape()))
    
        beta_a = _get_weights_wrapper(
                name='beta_a', shape=[1, shape[3]]
                )
        poses, activations = routing(u_hat_vecs, beta_a, iterations, shape[3], i_activations_patches)   # [2425,16,16]  [2425,16]
        poses = tf.reshape(poses, [
                    inputs_poses_shape[0], inputs_poses_shape[1],
                    inputs_poses_shape[2], shape[3],
                    inputs_poses_shape[-1]]
                ) 
        activations = tf.reshape(activations, [
                    inputs_poses_shape[0],inputs_poses_shape[1],
                    inputs_poses_shape[2],shape[3]]
                ) 
        nets = poses, activations      #25, 97, 1, 16, 16         25, 97, 1, 16
    tf.logging.info("capsule conv poses dimension:{}".format(poses.get_shape()))
    tf.logging.info("capsule conv activations dimension:{}".format(activations.get_shape()))
    return nets