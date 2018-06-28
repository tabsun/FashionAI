import sys
import math

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

import common

DEFAULT_PADDING = 'SAME'
USE_FUSED_BN = True
BN_EPSILON = 9.999999747378752e-06
BN_MOMENTUM = 0.99

_init_xavier = tf.contrib.layers.xavier_initializer()
_init_norm = tf.truncated_normal_initializer(stddev=0.01)
_init_zero = slim.init_ops.zeros_initializer()
_l2_regularizer_00004 = tf.contrib.layers.l2_regularizer(0.00004)
_l2_regularizer_convb = tf.contrib.layers.l2_regularizer(common.regularizer_conv)


def layer(op):
    '''
    Decorator for composable network layers.
    '''

    def layer_decorated(self, *args, **kwargs):
        # Automatically set a name if not provided.
        name = kwargs.setdefault('name', self.get_unique_name(op.__name__))
        # Figure out the layer inputs.
        if len(self.terminals) == 0:
            raise RuntimeError('No input variables found for layer %s.' % name)
        elif len(self.terminals) == 1:
            layer_input = self.terminals[0]
        else:
            layer_input = list(self.terminals)
        # Perform the operation and get the output.
        layer_output = op(self, layer_input, *args, **kwargs)
        # Add to layer LUT.
        self.layers[name] = layer_output
        # This output is now the input for the next layer.
        self.feed(layer_output)
        # Return self for chained calls.
        return self

    return layer_decorated


class BaseNetwork(object):
    def __init__(self, inputs, clothe_class='', trainable=True):
        # The input nodes for this network
        self.inputs = inputs
        # The current list of terminal nodes
        self.terminals = []
        # Mapping from layer names to layers
        self.layers = dict(inputs)
        # If true, the resulting variables are set as trainable
        self.trainable = trainable
        # Switch variable for dropout
        self.use_dropout = tf.placeholder_with_default(tf.constant(1.0),
                                                       shape=[],
                                                       name='use_dropout')
        # Each network has different output channels number
        self.clothe_class = clothe_class

        self.setup()

    def setup(self):
        '''Construct the network. '''
        raise NotImplementedError('Must be implemented by the subclass.')

    def load(self, data_path, session, ignore_missing=False):
        '''
        Load network weights.
        data_path: The path to the numpy-serialized network weights
        session: The current TensorFlow session
        ignore_missing: If true, serialized weights for missing layers are ignored.
        '''
        data_dict = np.load(data_path, encoding='bytes').item()
        for index, op_name in enumerate(data_dict.keys()):
            print op_name, "  {} / {}".format(index, len(data_dict.keys()))
            if 'conv' not in op_name:
                continue
            with tf.variable_scope('', reuse=True):
                try:
                    var = tf.get_variable(op_name.decode("utf-8"))
                    session.run(var.assign(data_dict[op_name]))
                except ValueError as e:
                    print(e)
                    if not ignore_missing:
                        raise

    def feed(self, *args):
        '''Set the input(s) for the next operation by replacing the terminal nodes.
        The arguments can be either layer names or the actual layers.
        '''
        assert len(args) != 0
        self.terminals = []
        for fed_layer in args:
            try:
                is_str = isinstance(fed_layer, basestring)
            except NameError:
                is_str = isinstance(fed_layer, str)
            if is_str:
                try:
                    fed_layer = self.layers[fed_layer]
                except KeyError:
                    raise KeyError('Unknown layer name fed: %s' % fed_layer)
            self.terminals.append(fed_layer)
        return self

    def get_output(self, name=None):
        '''Returns the current network output.'''
        if not name:
            return self.terminals[-1]
        else:
            return self.layers[name]

    def get_tensor(self, name):
        return self.get_output(name)

    def get_unique_name(self, prefix):
        '''Returns an index-suffixed unique name for the given prefix.
        This is used for auto-generating layer names based on the type-prefix.
        '''
        ident = sum(t.startswith(prefix) for t, _ in self.layers.items()) + 1
        return '%s_%d' % (prefix, ident)

    def make_var(self, name, shape, trainable=True):
        '''Creates a new TensorFlow variable.'''
        return tf.get_variable(name, shape, trainable=self.trainable & trainable, initializer=tf.contrib.layers.xavier_initializer())

    def validate_padding(self, padding):
        '''Verifies that the padding is one of the supported ones.'''
        assert padding in ('SAME', 'VALID')

    def get_bilinear_filter(self, filter_shape, upscale_factor, name):
        #with tf.variable_scope(name):
        ##filter_shape is [width, height, num_in_channels, num_out_channels]
        kernel_size = filter_shape[1]
        ### Centre location of the filter for which value is calculated
        if kernel_size % 2 == 1:
            centre_location = upscale_factor - 1
        else:
            centre_location = upscale_factor - 0.5
 
        bilinear = np.zeros([filter_shape[0], filter_shape[1]])
        for x in range(filter_shape[0]):
            for y in range(filter_shape[1]):
                ##Interpolation Calculation
                value = (1 - abs((x - centre_location)/ upscale_factor)) * (1 - abs((y - centre_location)/ upscale_factor))
                bilinear[x, y] = value
        weights = np.zeros(filter_shape)
        for i in range(filter_shape[2]):
            weights[:, :, i, i] = bilinear
        init = tf.constant_initializer(value=weights,
                                       dtype=tf.float32)
 
        bilinear_weights = tf.get_variable(name="decon_bilinear_filter", initializer=init,
                               shape=weights.shape)
        return bilinear_weights
    
    # input image order: BGR, range [0-255]
    # mean_value: 104, 117, 123
    # only subtract mean is used
    def constant_xavier_initializer(self, shape, group, dtype=tf.float32, uniform=True):
        """Initializer function."""
        if not dtype.is_floating:
            raise TypeError('Cannot create initializer for non-floating point type.')
        # Estimating fan_in and fan_out is not possible to do perfectly, but we try.
        # This is the right thing for matrix multiply and convolutions.
        if shape:
            fan_in = float(shape[-2]) if len(shape) > 1 else float(shape[-1])
            fan_out = float(shape[-1])/group
        else:
            fan_in = 1.0
            fan_out = 1.0
        for dim in shape[:-2]:
            fan_in *= float(dim)
            fan_out *= float(dim)
    
        # Average number of inputs and output connections.
        n = (fan_in + fan_out) / 2.0
        if uniform:
            # To get stddev = math.sqrt(factor / n) need to adjust for uniform.
            limit = math.sqrt(3.0 * 1.0 / n)
            return tf.random_uniform(shape, -limit, limit, dtype, seed=None)
        else:
            # To get stddev = math.sqrt(factor / n) need to adjust for truncated.
            trunc_stddev = math.sqrt(1.3 * 1.0 / n)
            return tf.truncated_normal(shape, 0.0, trunc_stddev, dtype, seed=None)

    @layer
    def sext_bottleneck_block(self, input, input_filters, is_training, group, data_format='channels_last', need_reduce=True, is_root=False, name=None, reduced_scale=16):
        bn_axis = -1 if data_format == 'channels_last' else 1
        strides_to_use = 1
        name_prefix = name
        residuals = input
        if need_reduce:
            strides_to_use = 1 if is_root else 2
            proj_mapping = tf.layers.conv2d(input, input_filters * 2, (1, 1), use_bias=False,
                                    name=name_prefix + '_1x1_proj', strides=(strides_to_use, strides_to_use),
                                    padding='valid', data_format=data_format, activation=None,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                    bias_initializer=tf.zeros_initializer())
            residuals = tf.layers.batch_normalization(proj_mapping, momentum=BN_MOMENTUM,
                                    name=name_prefix + '_1x1_proj/bn', axis=bn_axis,
                                    epsilon=BN_EPSILON, training=is_training, reuse=None, fused=USE_FUSED_BN)
        reduced_inputs = tf.layers.conv2d(input, input_filters, (1, 1), use_bias=False,
                                name=name_prefix + '_1x1_reduce', strides=(strides_to_use, strides_to_use),
                                padding='valid', data_format=data_format, activation=None,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                bias_initializer=tf.zeros_initializer())
        reduced_inputs_bn = tf.layers.batch_normalization(reduced_inputs, momentum=BN_MOMENTUM,
                                            name=name_prefix + '_1x1_reduce/bn', axis=bn_axis,
                                            epsilon=BN_EPSILON, training=is_training, reuse=None, fused=USE_FUSED_BN)
        reduced_inputs_relu = tf.nn.relu(reduced_inputs_bn, name=name_prefix + '_1x1_reduce/relu')

        # NeXt part - group operation
        if data_format == 'channels_first':
            reduced_inputs_relu = tf.pad(reduced_inputs_relu, paddings=[[0,0],[0,0],[1,1],[1,1]])
            weight_shape = [3, 3, reduced_inputs_relu.get_shape().as_list()[1]//group, input_filters]
            weight_ =tf.Variable(self.constant_xavier_initializer(weight_shape, group), trainable=is_training, name=name_prefix+'_3x3/kernel')
            weight_groups = tf.split(weight_, num_or_size_splits=group, axis=-1, name=name_prefix+'_weight_split')
            xs = tf.split(reduced_inputs_relu, num_or_size_splits=group, axis=1, name=name_prefix+'_inputs_split')
        else:
            reduced_inputs_relu = tf.pad(reduced_inputs_relu, paddings=[[0,0],[1,1],[1,1],[0,0]])
            weight_shape = [3, 3, reduced_inputs_relu.get_shape().as_list()[-1]//group, input_filters]
            weight_ = tf.Variable(self.constant_xavier_initializer(weight_shape, group), trainable=is_training, name=name_prefix+'_3x3/kernel')
            weight_groups = tf.split(weight_, num_or_size_splits=group, axis=-1, name=name_prefix+'_weight_split')
            xs = tf.split(reduced_inputs_relu, num_or_size_splits=group, axis=-1, name=name_prefix+'_inputs_split')

        convolved = [tf.nn.convolution(x, weight, padding='VALID', strides=[strides_to_use,strides_to_use], name=name_prefix+'_group_conv', 
                                       data_format=('NCHW' if data_format == 'channels_first' else 'NHWC')) for (x, weight) in zip(xs, weight_groups)]

        if data_format == 'channels_first':
            conv3_inputs = tf.concat(convolved, axis=1, name=name_prefix+'_concat')
        else:
            conv3_inputs = tf.concat(convolved, axis=-1, name=name_prefix+'_concat')

        conv3_inputs_bn = tf.layers.batch_normalization(conv3_inputs, momentum=BN_MOMENTUM, name=name_prefix + '_3x3/bn',
                                            axis=bn_axis, epsilon=BN_EPSILON, training=is_training, reuse=None, fused=USE_FUSED_BN)
        conv3_inputs_relu = tf.nn.relu(conv3_inputs_bn, name=name_prefix + '_3x3/relu')


        increase_inputs = tf.layers.conv2d(conv3_inputs_relu, input_filters * 2, (1, 1), use_bias=False,
                                    name=name_prefix + '_1x1_increase', strides=(1, 1),
                                    padding='valid', data_format=data_format, activation=None,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                    bias_initializer=tf.zeros_initializer())

        increase_inputs_bn = tf.layers.batch_normalization(increase_inputs, momentum=BN_MOMENTUM,
                                            name=name_prefix + '_1x1_increase/bn', axis=bn_axis,
                                            epsilon=BN_EPSILON, training=is_training, reuse=None, fused=USE_FUSED_BN)

        if data_format == 'channels_first':
            pooled_inputs = tf.reduce_mean(increase_inputs_bn, [2, 3], name=name_prefix + '_global_pool', keep_dims=True)
        else:
            pooled_inputs = tf.reduce_mean(increase_inputs_bn, [1, 2], name=name_prefix + '_global_pool', keep_dims=True)

        down_inputs = tf.layers.conv2d(pooled_inputs, (input_filters * 2) // reduced_scale, (1, 1), use_bias=True,
                                    name=name_prefix + '_1x1_down', strides=(1, 1),
                                    padding='valid', data_format=data_format, activation=None,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                    bias_initializer=tf.zeros_initializer())
        down_inputs_relu = tf.nn.relu(down_inputs, name=name_prefix + '_1x1_down/relu')


        up_inputs = tf.layers.conv2d(down_inputs_relu, input_filters * 2, (1, 1), use_bias=True,
                                    name=name_prefix + '_1x1_up', strides=(1, 1),
                                    padding='valid', data_format=data_format, activation=None,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                    bias_initializer=tf.zeros_initializer())
        prob_outputs = tf.nn.sigmoid(up_inputs, name=name_prefix + '_prob')

        rescaled_feat = tf.multiply(prob_outputs, increase_inputs_bn, name=name_prefix + '_mul')
        pre_act = tf.add(residuals, rescaled_feat, name=name_prefix + '_add')
        return tf.nn.relu(pre_act, name=name_prefix)
    # input image order: BGR, range [0-255]
    # mean_value: 104, 117, 123
    # only subtract mean is used
    # for root block, use dummy input_filters, e.g. 128 rather than 64 for the first block
    @layer
    def se_bottleneck_block(self, input, input_filters, is_training, data_format='channels_last', need_reduce=True, is_root=False, name=None, reduced_scale=16):
        bn_axis = -1 if data_format == 'channels_last' else 1
        strides_to_use = 1
        name_prefix = name
        residuals = input
        if need_reduce:
            strides_to_use = 1 if is_root else 2
            proj_mapping = tf.layers.conv2d(input, input_filters * 2, (1, 1), use_bias=False,
                                    name=name_prefix + '_1x1_proj', strides=(strides_to_use, strides_to_use),
                                    padding='valid', data_format=data_format, activation=None,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                    bias_initializer=tf.zeros_initializer())
            residuals = tf.layers.batch_normalization(proj_mapping, momentum=BN_MOMENTUM,
                                    name=name_prefix + '_1x1_proj/bn', axis=bn_axis,
                                    epsilon=BN_EPSILON, training=is_training, reuse=None, fused=USE_FUSED_BN)
        reduced_inputs = tf.layers.conv2d(input, input_filters / 2, (1, 1), use_bias=False,
                                name=name_prefix + '_1x1_reduce', strides=(strides_to_use, strides_to_use),
                                padding='valid', data_format=data_format, activation=None,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                bias_initializer=tf.zeros_initializer())
        reduced_inputs_bn = tf.layers.batch_normalization(reduced_inputs, momentum=BN_MOMENTUM,
                                            name=name_prefix + '_1x1_reduce/bn', axis=bn_axis,
                                            epsilon=BN_EPSILON, training=is_training, reuse=None, fused=USE_FUSED_BN)
        reduced_inputs_relu = tf.nn.relu(reduced_inputs_bn, name=name_prefix + '_1x1_reduce/relu')


        conv3_inputs = tf.layers.conv2d(reduced_inputs_relu, input_filters / 2, (3, 3), use_bias=False,
                                    name=name_prefix + '_3x3', strides=(1, 1),
                                    padding='same', data_format=data_format, activation=None,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                    bias_initializer=tf.zeros_initializer())
        conv3_inputs_bn = tf.layers.batch_normalization(conv3_inputs, momentum=BN_MOMENTUM, name=name_prefix + '_3x3/bn',
                                            axis=bn_axis, epsilon=BN_EPSILON, training=is_training, reuse=None, fused=USE_FUSED_BN)
        conv3_inputs_relu = tf.nn.relu(conv3_inputs_bn, name=name_prefix + '_3x3/relu')


        increase_inputs = tf.layers.conv2d(conv3_inputs_relu, input_filters * 2, (1, 1), use_bias=False,
                                    name=name_prefix + '_1x1_increase', strides=(1, 1),
                                    padding='valid', data_format=data_format, activation=None,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                    bias_initializer=tf.zeros_initializer())

        increase_inputs_bn = tf.layers.batch_normalization(increase_inputs, momentum=BN_MOMENTUM,
                                            name=name_prefix + '_1x1_increase/bn', axis=bn_axis,
                                            epsilon=BN_EPSILON, training=is_training, reuse=None, fused=USE_FUSED_BN)

        if data_format == 'channels_first':
            pooled_inputs = tf.reduce_mean(increase_inputs_bn, [2, 3], name=name_prefix + '_global_pool', keep_dims=True)
        else:
            pooled_inputs = tf.reduce_mean(increase_inputs_bn, [1, 2], name=name_prefix + '_global_pool', keep_dims=True)

        down_inputs = tf.layers.conv2d(pooled_inputs, (input_filters * 2) // reduced_scale, (1, 1), use_bias=True,
                                    name=name_prefix + '_1x1_down', strides=(1, 1),
                                    padding='valid', data_format=data_format, activation=None,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                    bias_initializer=tf.zeros_initializer())
        down_inputs_relu = tf.nn.relu(down_inputs, name=name_prefix + '_1x1_down/relu')


        up_inputs = tf.layers.conv2d(down_inputs_relu, input_filters * 2, (1, 1), use_bias=True,
                                    name=name_prefix + '_1x1_up', strides=(1, 1),
                                    padding='valid', data_format=data_format, activation=None,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                    bias_initializer=tf.zeros_initializer())
        prob_outputs = tf.nn.sigmoid(up_inputs, name=name_prefix + '_prob')

        rescaled_feat = tf.multiply(prob_outputs, increase_inputs_bn, name=name_prefix + '_mul')
        pre_act = tf.add(residuals, rescaled_feat, name=name_prefix + '_add')
        return tf.nn.relu(pre_act, name=name_prefix)
    
    @layer
    def conv_block(self, input, k_h, k_w, c_o, stride, data_format, is_training, name, use_bias=False ):
        bn_axis = -1 if data_format == 'channels_last' else 1
        pool_ind = name.split('conv')[-1]
        input = tf.layers.conv2d(input, c_o, (k_h, k_w), use_bias=use_bias,
                                name='{}/{}x{}_s{}'.format(name,k_h,k_w,stride), strides=(stride, stride),
                                padding='valid', data_format=data_format, activation=None,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                bias_initializer=tf.zeros_initializer())

        input = tf.layers.batch_normalization(input, momentum=BN_MOMENTUM,
                                            name='{}/{}x{}_s{}/bn'.format(name,k_h,k_w,stride), axis=bn_axis,
                                            epsilon=BN_EPSILON, training=is_training, reuse=None, fused=USE_FUSED_BN)
        input = tf.nn.relu(input, name='{}/relu_{}x{}_s{}'.format(name,k_h,k_w,stride))
        input = tf.layers.max_pooling2d(input, [3, 3], [2, 2], padding='same', data_format=data_format, name='pool{}/3x3_s2'.format(pool_ind))
        return input

    @layer
    def normalize_seres50(self, input, name, data_format='channels_last'):
        # convert from RGB to BGR
        if data_format == 'channels_last':
            channels = tf.unstack(input, axis=-1)
            swaped_input_image = tf.stack([tf.subtract(channels[2], 104.0, name='B_sub'), 
                                           tf.subtract(channels[1], 117.0, name='G_sub'),
                                           tf.subtract(channels[0], 123.0, name='R_sub')], axis=-1)
        else:
            channels = tf.unstack(input, axis=1)
            swaped_input_image = tf.stack([tf.subtract(channels[2], 104.0, name='B_sub'), 
                                           tf.subtract(channels[1], 117.0, name='G_sub'),
                                           tf.subtract(channels[0], 123.0, name='R_sub')], axis=1)

        if data_format == 'channels_first':
            swaped_input_image = tf.pad(swaped_input_image, paddings = [[0, 0], [0, 0], [3, 3], [3, 3]])
        else:
            swaped_input_image = tf.pad(swaped_input_image, paddings = [[0, 0], [3, 3], [3, 3], [0, 0]])
        return swaped_input_image

    @layer
    def normalize_vgg(self, input, name):
        # normalize input -1.0 ~ 1.0
        input = tf.divide(input, 255.0, name=name + '_divide')
        input = tf.subtract(input, 0.5, name=name + '_subtract')
        input = tf.multiply(input, 2.0, name=name + '_multiply')
        return input

    @layer
    def normalize_mobilenet(self, input, name):
        input = tf.divide(input, 255.0, name=name + '_divide')
        input = tf.subtract(input, 0.5, name=name + '_subtract')
        input = tf.multiply(input, 2.0, name=name + '_multiply')
        return input

    @layer
    def normalize_nasnet(self, input, name):
        input = tf.divide(input, 255.0, name=name + '_divide')
        input = tf.subtract(input, 0.5, name=name + '_subtract')
        input = tf.multiply(input, 2.0, name=name + '_multiply')
        return input

    @layer
    def upsample(self, input, name, factor=None, target_size=None):
        if factor:
            return tf.image.resize_bilinear(input, [int(int(input.get_shape()[1]) * factor), int(int(input.get_shape()[2]) * factor)], name=name)
        if target_size:
            return tf.image.resize_bilinear(input, [target_size, target_size], name=name)

    @layer
    def separable_conv(self, input, k_h, k_w, c_o, stride, name, relu=True, set_bias=True):
        with slim.arg_scope([slim.batch_norm], decay=0.999, fused=common.batchnorm_fused, is_training=self.trainable):
            output = slim.separable_convolution2d(input,
                                                  num_outputs=None,
                                                  stride=stride,
                                                  trainable=self.trainable,
                                                  depth_multiplier=1.0,
                                                  kernel_size=[k_h, k_w],
                                                  # activation_fn=common.activation_fn if relu else None,
                                                  activation_fn=None,
                                                  # normalizer_fn=slim.batch_norm,
                                                  weights_initializer=_init_xavier,
                                                  # weights_initializer=_init_norm,
                                                  weights_regularizer=_l2_regularizer_00004,
                                                  biases_initializer=None,
                                                  padding=DEFAULT_PADDING,
                                                  scope=name + '_depthwise')

            output = slim.convolution2d(output,
                                        c_o,
                                        stride=1,
                                        kernel_size=[1, 1],
                                        activation_fn=common.activation_fn if relu else None,
                                        weights_initializer=_init_xavier,
                                        # weights_initializer=_init_norm,
                                        biases_initializer=_init_zero if set_bias else None,
                                        normalizer_fn=slim.batch_norm,
                                        trainable=self.trainable,
                                        weights_regularizer=None,
                                        scope=name + '_pointwise')

        return output

    @layer
    def convb(self, input, k_h, k_w, c_o, stride, name, relu=True, set_bias=True, set_tanh=False):
        with slim.arg_scope([slim.batch_norm], decay=0.999, fused=common.batchnorm_fused, is_training=self.trainable):
            output = slim.convolution2d(input, c_o, kernel_size=[k_h, k_w],
                                        stride=stride,
                                        normalizer_fn=slim.batch_norm,
                                        weights_regularizer=_l2_regularizer_convb,
                                        weights_initializer=_init_xavier,
                                        # weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                        biases_initializer=_init_zero if set_bias else None,
                                        trainable=self.trainable,
                                        activation_fn=common.activation_fn if relu else None,
                                        scope=name)
            if set_tanh:
                output = tf.nn.sigmoid(output, name=name + '_extra_acv')
        return output

    @layer
    def conv(self,
             input,
             k_h,
             k_w,
             c_o,
             s_h,
             s_w,
             name,
             relu=True,
             padding=DEFAULT_PADDING,
             group=1,
             trainable=True,
             biased=True):
        # Verify that the padding is acceptable
        self.validate_padding(padding)
        # Get the number of channels in the input
        c_i = int(input.get_shape()[-1])
        # Verify that the grouping parameter is valid
        assert c_i % group == 0
        assert c_o % group == 0
        # Convolution for a given input and kernel
        convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
        with tf.variable_scope(name) as scope:
            kernel = self.make_var('weights', shape=[k_h, k_w, c_i / group, c_o], trainable=self.trainable & trainable)
            if group == 1:
                # This is the common-case. Convolve the input without any further complications.
                output = convolve(input, kernel)
            else:
                # Split the input into groups and then convolve each of them independently
                input_groups = tf.split(3, group, input)
                kernel_groups = tf.split(3, group, kernel)
                output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
                # Concatenate the groups
                output = tf.concat(3, output_groups)
            # Add the biases
            if biased:
                biases = self.make_var('biases', [c_o], trainable=self.trainable & trainable)
                output = tf.nn.bias_add(output, biases)

            if relu:
                # ReLU non-linearity
                output = tf.nn.relu(output, name=scope.name)
            return output

    @layer
    def deconv(self, input, output_channels, upscale_factor, name):
        kernel_size = 2*upscale_factor - upscale_factor%2
        stride = upscale_factor
        strides = [1, stride, stride, 1]

        input_shape = tf.shape(input)
        h = input_shape[1] * stride
        w = input_shape[2] * stride
        #
        output_shape = tf.stack([input_shape[0], h, w, output_channels])
        filter_shape = [kernel_size, kernel_size, output_channels, output_channels]
        ##W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1))
        W = self.get_bilinear_filter(filter_shape, upscale_factor, name)
        
        #pad_space_h = (input_shape[1] * upscale_factor - h)
        #pad_space_w = (input_shape[2] * upscale_factor - w)
        #upsample = tf.nn.conv2d_transpose(input, W, output_shape, strides=strides, padding='SAME')
        #paddings = tf.constant([[0,0],[0,1],[0,1],[0,0]])
        
        return tf.nn.conv2d_transpose(input, W, output_shape, strides=strides, padding='SAME', name=name) #tf.pad(upsample, paddings, 'CONSTANT')

    @layer
    def relu(self, input, name):
        return tf.nn.relu(input, name=name)

    @layer
    def max_pool(self, input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
        self.validate_padding(padding)
        return tf.nn.max_pool(input,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name)

    @layer
    def avg_pool(self, input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
        self.validate_padding(padding)
        return tf.nn.avg_pool(input,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name)

    @layer
    def lrn(self, input, radius, alpha, beta, name, bias=1.0):
        return tf.nn.local_response_normalization(input,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias,
                                                  name=name)

    @layer
    def concat(self, inputs, axis, name):
        return tf.concat(axis=axis, values=inputs, name=name)

    @layer
    def add(self, inputs, name):
        return tf.add_n(inputs, name=name)

    @layer
    def fc(self, input, num_out, name, relu=True):
        with tf.variable_scope(name) as scope:
            input_shape = input.get_shape()
            if input_shape.ndims == 4:
                # The input is spatial. Vectorize it first.
                dim = 1
                for d in input_shape[1:].as_list():
                    dim *= d
                feed_in = tf.reshape(input, [-1, dim])
            else:
                feed_in, dim = (input, input_shape[-1].value)
            weights = self.make_var('weights', shape=[dim, num_out])
            biases = self.make_var('biases', [num_out])
            op = tf.nn.relu_layer if relu else tf.nn.xw_plus_b
            fc = op(feed_in, weights, biases, name=scope.name)
            return fc

    @layer
    def softmax(self, input, name):
        input_shape = map(lambda v: v.value, input.get_shape())
        if len(input_shape) > 2:
            # For certain models (like NiN), the singleton spatial dimensions
            # need to be explicitly squeezed, since they're not broadcast-able
            # in TensorFlow's NHWC ordering (unlike Caffe's NCHW).
            if input_shape[1] == 1 and input_shape[2] == 1:
                input = tf.squeeze(input, squeeze_dims=[1, 2])
            else:
                raise ValueError('Rank 2 tensor input expected for softmax!')
        return tf.nn.softmax(input, name=name)

    @layer
    def batch_normalization(self, input, name, scale_offset=True, relu=False):
        # NOTE: Currently, only inference is supported
        with tf.variable_scope(name) as scope:
            shape = [input.get_shape()[-1]]
            if scale_offset:
                scale = self.make_var('scale', shape=shape)
                offset = self.make_var('offset', shape=shape)
            else:
                scale, offset = (None, None)
            output = tf.nn.batch_normalization(
                input,
                mean=self.make_var('mean', shape=shape),
                variance=self.make_var('variance', shape=shape),
                offset=offset,
                scale=scale,
                # TODO: This is the default Caffe batch norm eps
                # Get the actual eps from parameters
                variance_epsilon=1e-5,
                name=name)
            if relu:
                output = tf.nn.relu(output)
            return output

    @layer
    def dropout(self, input, keep_prob, name):
        keep = 1 - self.use_dropout + (self.use_dropout * keep_prob)
        return tf.nn.dropout(input, keep, name=name)
