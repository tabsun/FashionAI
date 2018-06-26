import network_base
import tensorflow as tf

# TODO
# blouse      14 pts + 14 links
# outwear     15 pts + 15 links
# dress       16 pts + 16/17 links
# trousers    8 pts + 7/8/9 links
# skirt       5 pts + 4 links
# outwear2nd     apl apr wll wlr thl thr
output_channels = 16
channels_num = dict({'blouse': 14, 'outwear': 15, 'dress': 16, 'trousers': 8, 'skirt': 5,'outwear2nd':7})

class SEResNet50Network(network_base.BaseNetwork):
    def setup(self):
        # TODO
        data_format = 'channels_last'
        is_training = self.trainable
        if self.clothe_class in channels_num.keys():
            output_channels = channels_num[self.clothe_class]
        # Net structure
        (self.feed('image')
             .normalize_seres50(data_format=data_format, name='preprocess')
             # conv1
             .conv_block(7, 7, 64, 2, data_format, is_training, name='conv1')
             # conv2 x 3
             .se_bottleneck_block(128, is_training, data_format, True, True   ,name='conv2_1')
             .se_bottleneck_block(128, is_training, data_format, False, False ,name='conv2_2')
             .se_bottleneck_block(128, is_training, data_format, False, False ,name='conv2_3')
             # conv3 x 4                                                                     
             .se_bottleneck_block(256, is_training, data_format, True, True   ,name='conv3_1')
             .se_bottleneck_block(256, is_training, data_format, False, False ,name='conv3_2')
             .se_bottleneck_block(256, is_training, data_format, False, False ,name='conv3_3')
             .se_bottleneck_block(256, is_training, data_format, False, False ,name='conv3_4')
             # conv4 x 6                                                                     
             .se_bottleneck_block(512, is_training, data_format, True, True   ,name='conv4_1')
             .se_bottleneck_block(512, is_training, data_format, False, False ,name='conv4_2')
             .se_bottleneck_block(512, is_training, data_format, False, False ,name='conv4_3')
             .se_bottleneck_block(512, is_training, data_format, False, False ,name='conv4_4')
             .se_bottleneck_block(512, is_training, data_format, False, False ,name='conv4_5')
             .se_bottleneck_block(512, is_training, data_format, False, False ,name='conv4_6')
             # conv5 x 3                                                                      
             .se_bottleneck_block(1024, is_training, data_format, True, True  , name='conv5_1')
             .se_bottleneck_block(1024, is_training, data_format, False, False, name='conv5_2')
             .se_bottleneck_block(1024, is_training, data_format, False, False, name='conv5_3'))
                     
        (self.feed('conv2_3')
             .upsample(target_size=46, name='f_conv2_3'))
        (self.feed('conv3_4')
             .upsample(target_size=46, name='f_conv3_4'))
        (self.feed('conv4_6')
             .upsample(target_size=46, name='f_conv4_6'))
        (self.feed('conv5_3')
             .upsample(target_size=46, name='f_conv5_3'))

        (self.feed('f_conv2_3',
                   'f_conv3_4',
                   'f_conv4_6',
                   'f_conv5_3')
             .concat(3, name='concat_f')
             .conv(3, 3, 512, 1, 1, name='conv5_4_CPM')
             .conv(3, 3, 128, 1, 1, name='conv5_5_CPM') #******

             .conv(3, 3, 128, 1, 1, name='conv5_6_CPM_L1')
             .conv(3, 3, 128, 1, 1, name='conv5_7_CPM_L1')
             .conv(3, 3, 128, 1, 1, name='conv5_8_CPM_L1')
             .conv(1, 1, 512, 1, 1, name='conv5_9_CPM_L1')
             .conv(1, 1, output_channels*2, 1, 1, relu=False, name='conv5_10_CPM_L1')) 

        (self.feed('conv5_5_CPM')
             .conv(3, 3, 128, 1, 1, name='conv5_6_CPM_L2')
             .conv(3, 3, 128, 1, 1, name='conv5_7_CPM_L2')
             .conv(3, 3, 128, 1, 1, name='conv5_8_CPM_L2')
             .conv(1, 1, 512, 1, 1, name='conv5_9_CPM_L2')
             .conv(1, 1, output_channels, 1, 1, relu=False, name='conv5_10_CPM_L2')) 

        (self.feed('conv5_10_CPM_L1',
                   'conv5_10_CPM_L2',
                   'conv5_5_CPM')
             .concat(3, name='concat_stage2')
             .conv(7, 7, 128, 1, 1, name='Mconv1_stage2_L1')
             .conv(7, 7, 128, 1, 1, name='Mconv2_stage2_L1')
             .conv(7, 7, 128, 1, 1, name='Mconv3_stage2_L1')
             .conv(7, 7, 128, 1, 1, name='Mconv4_stage2_L1')
             .conv(7, 7, 128, 1, 1, name='Mconv5_stage2_L1')
             .conv(1, 1, 128, 1, 1, name='Mconv6_stage2_L1')
             .conv(1, 1, output_channels*2, 1, 1, relu=False, name='Mconv7_stage2_L1'))

        (self.feed('concat_stage2')
             .conv(7, 7, 128, 1, 1, name='Mconv1_stage2_L2')
             .conv(7, 7, 128, 1, 1, name='Mconv2_stage2_L2')
             .conv(7, 7, 128, 1, 1, name='Mconv3_stage2_L2')
             .conv(7, 7, 128, 1, 1, name='Mconv4_stage2_L2')
             .conv(7, 7, 128, 1, 1, name='Mconv5_stage2_L2')
             .conv(1, 1, 128, 1, 1, name='Mconv6_stage2_L2')
             .conv(1, 1, output_channels, 1, 1, relu=False, name='Mconv7_stage2_L2'))

        (self.feed('Mconv7_stage2_L1',
                   'Mconv7_stage2_L2',
                   'conv5_5_CPM')
             .concat(3, name='concat_stage3')
             .conv(7, 7, 128, 1, 1, name='Mconv1_stage3_L1')
             .conv(7, 7, 128, 1, 1, name='Mconv2_stage3_L1')
             .conv(7, 7, 128, 1, 1, name='Mconv3_stage3_L1')
             .conv(7, 7, 128, 1, 1, name='Mconv4_stage3_L1')
             .conv(7, 7, 128, 1, 1, name='Mconv5_stage3_L1')
             .conv(1, 1, 128, 1, 1, name='Mconv6_stage3_L1')
             .conv(1, 1, output_channels*2, 1, 1, relu=False, name='Mconv7_stage3_L1'))

        (self.feed('concat_stage3')
             .conv(7, 7, 128, 1, 1, name='Mconv1_stage3_L2')
             .conv(7, 7, 128, 1, 1, name='Mconv2_stage3_L2')
             .conv(7, 7, 128, 1, 1, name='Mconv3_stage3_L2')
             .conv(7, 7, 128, 1, 1, name='Mconv4_stage3_L2')
             .conv(7, 7, 128, 1, 1, name='Mconv5_stage3_L2')
             .conv(1, 1, 128, 1, 1, name='Mconv6_stage3_L2')
             .conv(1, 1, output_channels, 1, 1, relu=False, name='Mconv7_stage3_L2'))

        (self.feed('Mconv7_stage3_L1',
                   'Mconv7_stage3_L2',
                   'conv5_5_CPM')
             .concat(3, name='concat_stage4')
             .conv(7, 7, 128, 1, 1, name='Mconv1_stage4_L1')
             .conv(7, 7, 128, 1, 1, name='Mconv2_stage4_L1')
             .conv(7, 7, 128, 1, 1, name='Mconv3_stage4_L1')
             .conv(7, 7, 128, 1, 1, name='Mconv4_stage4_L1')
             .conv(7, 7, 128, 1, 1, name='Mconv5_stage4_L1')
             .conv(1, 1, 128, 1, 1, name='Mconv6_stage4_L1')
             .conv(1, 1, output_channels*2, 1, 1, relu=False, name='Mconv7_stage4_L1'))

        (self.feed('concat_stage4')
             .conv(7, 7, 128, 1, 1, name='Mconv1_stage4_L2')
             .conv(7, 7, 128, 1, 1, name='Mconv2_stage4_L2')
             .conv(7, 7, 128, 1, 1, name='Mconv3_stage4_L2')
             .conv(7, 7, 128, 1, 1, name='Mconv4_stage4_L2')
             .conv(7, 7, 128, 1, 1, name='Mconv5_stage4_L2')
             .conv(1, 1, 128, 1, 1, name='Mconv6_stage4_L2')
             .conv(1, 1, output_channels, 1, 1, relu=False, name='Mconv7_stage4_L2'))

        (self.feed('Mconv7_stage4_L1',
                   'Mconv7_stage4_L2',
                   'conv5_5_CPM')
             .concat(3, name='concat_stage5')
             .conv(7, 7, 128, 1, 1, name='Mconv1_stage5_L1')
             .conv(7, 7, 128, 1, 1, name='Mconv2_stage5_L1')
             .conv(7, 7, 128, 1, 1, name='Mconv3_stage5_L1')
             .conv(7, 7, 128, 1, 1, name='Mconv4_stage5_L1')
             .conv(7, 7, 128, 1, 1, name='Mconv5_stage5_L1')
             .conv(1, 1, 128, 1, 1, name='Mconv6_stage5_L1')
             .conv(1, 1, output_channels*2, 1, 1, relu=False, name='Mconv7_stage5_L1'))

        (self.feed('concat_stage5')
             .conv(7, 7, 128, 1, 1, name='Mconv1_stage5_L2')
             .conv(7, 7, 128, 1, 1, name='Mconv2_stage5_L2')
             .conv(7, 7, 128, 1, 1, name='Mconv3_stage5_L2')
             .conv(7, 7, 128, 1, 1, name='Mconv4_stage5_L2')
             .conv(7, 7, 128, 1, 1, name='Mconv5_stage5_L2')
             .conv(1, 1, 128, 1, 1, name='Mconv6_stage5_L2')
             .conv(1, 1, output_channels, 1, 1, relu=False, name='Mconv7_stage5_L2'))

        (self.feed('Mconv7_stage5_L1',
                   'Mconv7_stage5_L2',
                   'conv5_5_CPM')
             .concat(3, name='concat_stage6')
             .conv(7, 7, 128, 1, 1, name='Mconv1_stage6_L1')
             .conv(7, 7, 128, 1, 1, name='Mconv2_stage6_L1')
             .conv(7, 7, 128, 1, 1, name='Mconv3_stage6_L1')
             .conv(7, 7, 128, 1, 1, name='Mconv4_stage6_L1')
             .conv(7, 7, 128, 1, 1, name='Mconv5_stage6_L1')
             .conv(1, 1, 128, 1, 1, name='Mconv6_stage6_L1')
             .conv(1, 1, output_channels*2, 1, 1, relu=False, name='Mconv7_stage6_L1'))

        (self.feed('concat_stage6')
             .conv(7, 7, 128, 1, 1, name='Mconv1_stage6_L2')
             .conv(7, 7, 128, 1, 1, name='Mconv2_stage6_L2')
             .conv(7, 7, 128, 1, 1, name='Mconv3_stage6_L2')
             .conv(7, 7, 128, 1, 1, name='Mconv4_stage6_L2')
             .conv(7, 7, 128, 1, 1, name='Mconv5_stage6_L2')
             .conv(1, 1, 128, 1, 1, name='Mconv6_stage6_L2')
             .conv(1, 1, output_channels, 1, 1, relu=False, name='Mconv7_stage6_L2'))

        with tf.variable_scope('Openpose'):
            (self.feed('Mconv7_stage6_L2',
                       'Mconv7_stage6_L1')
                 .concat(3, name='concat_stage7'))

    def loss_l1_l2(self):
         l1s = []
         l2s = []
         #l3s = []
         for layer_name in self.layers.keys():
              if 'Mconv7' in layer_name and '_L1' in layer_name:
                   l1s.append(self.layers[layer_name])
              if 'Mconv7' in layer_name and '_L2' in layer_name:
                   l2s.append(self.layers[layer_name])
              #if 'Mconv7' in layer_name and '_L3' in layer_name:
              #     l3s.append(self.layers[layer_name])

         return l1s, l2s #, l3s

    def loss_last(self):
         return self.get_output('Mconv7_stage6_L1'), self.get_output('Mconv7_stage6_L2') #, self.get_output('Mconv7_stage6_L3')

    def restorable_variables(self):
         return None
