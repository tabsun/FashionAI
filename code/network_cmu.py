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
class CmuNetwork(network_base.BaseNetwork):
    def setup(self):
        if self.clothe_class in channels_num.keys():
            output_channels = channels_num[self.clothe_class]
        (self.feed('image')
             .normalize_vgg(name='preprocess')
             .conv(3, 3, 64, 1, 1, name='conv1_1')
             .conv(3, 3, 64, 1, 1, name='conv1_2')
             .max_pool(2, 2, 2, 2, name='pool1_stage1')
             .conv(3, 3, 128, 1, 1, name='conv2_1')
             .conv(3, 3, 128, 1, 1, relu=False, name='conv2_2')
             .relu(name='conv2_2')
             .max_pool(2, 2, 2, 2, name='pool2_stage1')
             .conv(3, 3, 256, 1, 1, name='conv3_1')
             .conv(3, 3, 256, 1, 1, name='conv3_2')
             .conv(3, 3, 256, 1, 1, name='conv3_3')
             .conv(3, 3, 256, 1, 1, relu=False, name='conv3_4')
             .relu(name='conv3_4')
             .max_pool(2, 2, 2, 2, name='pool3_stage1')
             .conv(3, 3, 512, 1, 1, name='conv4_1')
             .conv(3, 3, 512, 1, 1, name='conv4_2')
             .conv(3, 3, 256, 1, 1, name='conv4_3_CPM')
             .conv(3, 3, 128, 1, 1, relu=False, name='conv4_4_CPM')
             .relu(name='conv4_4_CPM'))          # *****
             # experiment upsampling
        (self.feed('conv2_2')
            .upsample(0.25, name='feature_2_upsample'))
        (self.feed('conv3_4')
            .upsample(0.5, name='feature_3_upsample'))
             #.deconv(128, 2, name='conv4_4_CPM')
             #.upsample(2, name='conv4_4_CPM')

             #.upsample(4, name='upsample_conv4_4_CPM'))
             #.deconv2(128, 2, name='upsample1_conv4_4_CPM')
             #.deconv2(128, 2, name='upsample2_conv4_4_CPM')
             #.deconv2(128, 2, name='upsample_conv4_4_CPM'))

        (self.feed('conv4_4_CPM',
                   'feature_2_upsample',
                   'feature_3_upsample')
             .concat(3, name='concat_features')
             .conv(3, 3, 128, 1, 1, name='conv5_1_CPM_L1')
             .conv(3, 3, 128, 1, 1, name='conv5_2_CPM_L1')
             .conv(3, 3, 128, 1, 1, name='conv5_3_CPM_L1')
             .conv(1, 1, 512, 1, 1, name='conv5_4_CPM_L1')
             .conv(1, 1, output_channels*2, 1, 1, relu=False, name='conv5_5_CPM_L1'))

        (self.feed('concat_features')
             .conv(3, 3, 128, 1, 1, name='conv5_1_CPM_L2')
             .conv(3, 3, 128, 1, 1, name='conv5_2_CPM_L2')
             .conv(3, 3, 128, 1, 1, name='conv5_3_CPM_L2')
             .conv(1, 1, 512, 1, 1, name='conv5_4_CPM_L2')
             .conv(1, 1, output_channels, 1, 1, relu=False, name='conv5_5_CPM_L2'))

        #(self.feed('conv4_4_CPM')
        #     .conv(3, 3, 128, 1, 1, name='conv5_1_CPM_L3')
        #     .conv(3, 3, 128, 1, 1, name='conv5_2_CPM_L3')
        #     .conv(3, 3, 128, 1, 1, name='conv5_3_CPM_L3')
        #     .conv(1, 1, 512, 1, 1, name='conv5_4_CPM_L3')
        #     .conv(1, 1, output_channels*2, 1, 1, relu=False, name='conv5_5_CPM_L3'))

        (self.feed('conv5_5_CPM_L1',
                   'conv5_5_CPM_L2',
        #           'conv5_5_CPM_L3',
                   'concat_features')
             .concat(3, name='concat_stage2')
             .conv(7, 7, 128, 1, 1, name='Mconv1_stage2_L1')
             .conv(7, 7, 128, 1, 1, name='Mconv2_stage2_L1')
             .conv(7, 7, 128, 1, 1, name='Mconv3_stage2_L1')
             .conv(7, 7, 128, 1, 1, name='Mconv4_stage2_L1')
             .conv(7, 7, 128, 1, 1, name='Mconv5_stage2_L1')
             .conv(1, 1, 128, 1, 1, name='Mconv6_stage2_L1')
             .conv(1, 1, output_channels*2, 1, 1, relu=False, name='Mconv7_stage2_L1'))
             #.upsample(8, name='Mconv7_stage2_L1'))
             #.deconv2(output_channels*2, 2, name='upsample1_stage2_L1')
             #.deconv2(output_channels*2, 2, name='upsample2_stage2_L1')
             #.deconv2(output_channels*2, 2, name='Mconv7_stage2_L1'))

        (self.feed('concat_stage2')
             .conv(7, 7, 128, 1, 1, name='Mconv1_stage2_L2')
             .conv(7, 7, 128, 1, 1, name='Mconv2_stage2_L2')
             .conv(7, 7, 128, 1, 1, name='Mconv3_stage2_L2')
             .conv(7, 7, 128, 1, 1, name='Mconv4_stage2_L2')
             .conv(7, 7, 128, 1, 1, name='Mconv5_stage2_L2')
             .conv(1, 1, 128, 1, 1, name='Mconv6_stage2_L2')
             .conv(1, 1, output_channels, 1, 1, relu=False, name='Mconv7_stage2_L2'))
             #.upsample(8, name='Mconv7_stage2_L2'))
             #.deconv2(output_channels, 2, name='upsample1_stage2_L2')
             #.deconv2(output_channels, 2, name='upsample2_stage2_L2')
             #.deconv2(output_channels, 2, name='Mconv7_stage2_L2'))

        # (self.feed('concat_stage2')
        #      .conv(7, 7, 128, 1, 1, name='Mconv1_stage2_L3')
        #      .conv(7, 7, 128, 1, 1, name='Mconv2_stage2_L3')
        #      .conv(7, 7, 128, 1, 1, name='Mconv3_stage2_L3')
        #      .conv(7, 7, 128, 1, 1, name='Mconv4_stage2_L3')
        #      .conv(7, 7, 128, 1, 1, name='Mconv5_stage2_L3')
        #      .conv(1, 1, 128, 1, 1, name='Mconv6_stage2_L3')
        #      .conv(1, 1, output_channels*2, 1, 1, relu=False, name='Mconv7_stage2_L3'))


        (self.feed('Mconv7_stage2_L1',
                   'Mconv7_stage2_L2',
        #           'Mconv7_stage2_L3',
                   'concat_features')
             .concat(3, name='concat_stage3')
             .conv(7, 7, 128, 1, 1, name='Mconv1_stage3_L1')
             .conv(7, 7, 128, 1, 1, name='Mconv2_stage3_L1')
             .conv(7, 7, 128, 1, 1, name='Mconv3_stage3_L1')
             .conv(7, 7, 128, 1, 1, name='Mconv4_stage3_L1')
             .conv(7, 7, 128, 1, 1, name='Mconv5_stage3_L1')
             .conv(1, 1, 128, 1, 1, name='Mconv6_stage3_L1')
             .conv(1, 1, output_channels*2, 1, 1, relu=False, name='Mconv7_stage3_L1'))
             #.upsample(8, name='Mconv7_stage3_L1'))
             #.deconv2(output_channels*2, 2, name='upsample1_stage3_L1')
             #.deconv2(output_channels*2, 2, name='upsample2_stage3_L1')
             #.deconv2(output_channels*2, 2, name='Mconv7_stage3_L1'))


        (self.feed('concat_stage3')
             .conv(7, 7, 128, 1, 1, name='Mconv1_stage3_L2')
             .conv(7, 7, 128, 1, 1, name='Mconv2_stage3_L2')
             .conv(7, 7, 128, 1, 1, name='Mconv3_stage3_L2')
             .conv(7, 7, 128, 1, 1, name='Mconv4_stage3_L2')
             .conv(7, 7, 128, 1, 1, name='Mconv5_stage3_L2')
             .conv(1, 1, 128, 1, 1, name='Mconv6_stage3_L2')
             .conv(1, 1, output_channels, 1, 1, relu=False, name='Mconv7_stage3_L2'))
             #.upsample(8, name='Mconv7_stage3_L2'))
             #.deconv2(output_channels, 2, name='upsample1_stage3_L2')
             #.deconv2(output_channels, 2, name='upsample2_stage3_L2')
             #.deconv2(output_channels, 2, name='Mconv7_stage3_L2'))

        # (self.feed('concat_stage3')
        #      .conv(7, 7, 128, 1, 1, name='Mconv1_stage3_L3')
        #      .conv(7, 7, 128, 1, 1, name='Mconv2_stage3_L3')
        #      .conv(7, 7, 128, 1, 1, name='Mconv3_stage3_L3')
        #      .conv(7, 7, 128, 1, 1, name='Mconv4_stage3_L3')
        #      .conv(7, 7, 128, 1, 1, name='Mconv5_stage3_L3')
        #      .conv(1, 1, 128, 1, 1, name='Mconv6_stage3_L3')
        #      .conv(1, 1, output_channels*2, 1, 1, relu=False, name='Mconv7_stage3_L3'))

        (self.feed('Mconv7_stage3_L1',
                   'Mconv7_stage3_L2',
        #           'Mconv7_stage3_L3',
                   'concat_features')
             .concat(3, name='concat_stage4')
             .conv(7, 7, 128, 1, 1, name='Mconv1_stage4_L1')
             .conv(7, 7, 128, 1, 1, name='Mconv2_stage4_L1')
             .conv(7, 7, 128, 1, 1, name='Mconv3_stage4_L1')
             .conv(7, 7, 128, 1, 1, name='Mconv4_stage4_L1')
             .conv(7, 7, 128, 1, 1, name='Mconv5_stage4_L1')
             .conv(1, 1, 128, 1, 1, name='Mconv6_stage4_L1')
             .conv(1, 1, output_channels*2, 1, 1, relu=False, name='Mconv7_stage4_L1'))
             #.upsample(8, name='Mconv7_stage4_L1'))
             #.deconv2(output_channels*2, 2, name='upsample1_stage4_L1')
             #.deconv2(output_channels*2, 2, name='upsample2_stage4_L1')
             #.deconv2(output_channels*2, 2, name='Mconv7_stage4_L1'))

        (self.feed('concat_stage4')
             .conv(7, 7, 128, 1, 1, name='Mconv1_stage4_L2')
             .conv(7, 7, 128, 1, 1, name='Mconv2_stage4_L2')
             .conv(7, 7, 128, 1, 1, name='Mconv3_stage4_L2')
             .conv(7, 7, 128, 1, 1, name='Mconv4_stage4_L2')
             .conv(7, 7, 128, 1, 1, name='Mconv5_stage4_L2')
             .conv(1, 1, 128, 1, 1, name='Mconv6_stage4_L2')
             .conv(1, 1, output_channels, 1, 1, relu=False, name='Mconv7_stage4_L2'))
             #.upsample(8, name='Mconv7_stage4_L2'))
             #.deconv2(output_channels, 2, name='upsample1_stage4_L2')
             #.deconv2(output_channels, 2, name='upsample2_stage4_L2')
             #.deconv2(output_channels, 2, name='Mconv7_stage4_L2'))
        # (self.feed('concat_stage4')
        #      .conv(7, 7, 128, 1, 1, name='Mconv1_stage4_L3')
        #      .conv(7, 7, 128, 1, 1, name='Mconv2_stage4_L3')
        #      .conv(7, 7, 128, 1, 1, name='Mconv3_stage4_L3')
        #      .conv(7, 7, 128, 1, 1, name='Mconv4_stage4_L3')
        #      .conv(7, 7, 128, 1, 1, name='Mconv5_stage4_L3')
        #      .conv(1, 1, 128, 1, 1, name='Mconv6_stage4_L3')
        #      .conv(1, 1, output_channels*2, 1, 1, relu=False, name='Mconv7_stage4_L3'))

        (self.feed('Mconv7_stage4_L1',
                   'Mconv7_stage4_L2',
        #           'Mconv7_stage4_L3',
                   'concat_features')
             .concat(3, name='concat_stage5')
             .conv(7, 7, 128, 1, 1, name='Mconv1_stage5_L1')
             .conv(7, 7, 128, 1, 1, name='Mconv2_stage5_L1')
             .conv(7, 7, 128, 1, 1, name='Mconv3_stage5_L1')
             .conv(7, 7, 128, 1, 1, name='Mconv4_stage5_L1')
             .conv(7, 7, 128, 1, 1, name='Mconv5_stage5_L1')
             .conv(1, 1, 128, 1, 1, name='Mconv6_stage5_L1')
             .conv(1, 1, output_channels*2, 1, 1, relu=False, name='Mconv7_stage5_L1'))
             #.upsample(8, name='Mconv7_stage5_L1'))
             #.deconv2(output_channels*2, 2, name='upsample1_stage5_L1')
             #.deconv2(output_channels*2, 2, name='upsample2_stage5_L1')
             #.deconv2(output_channels*2, 2, name='Mconv7_stage5_L1'))

        (self.feed('concat_stage5')
             .conv(7, 7, 128, 1, 1, name='Mconv1_stage5_L2')
             .conv(7, 7, 128, 1, 1, name='Mconv2_stage5_L2')
             .conv(7, 7, 128, 1, 1, name='Mconv3_stage5_L2')
             .conv(7, 7, 128, 1, 1, name='Mconv4_stage5_L2')
             .conv(7, 7, 128, 1, 1, name='Mconv5_stage5_L2')
             .conv(1, 1, 128, 1, 1, name='Mconv6_stage5_L2')
             .conv(1, 1, output_channels, 1, 1, relu=False, name='Mconv7_stage5_L2'))
             #.upsample(8, name='Mconv7_stage5_L2'))
             #.deconv2(output_channels, 2, name='upsample1_stage5_L2')
             #.deconv2(output_channels, 2, name='upsample2_stage5_L2')
             #.deconv2(output_channels, 2, name='Mconv7_stage5_L2'))
        # (self.feed('concat_stage5')
        #      .conv(7, 7, 128, 1, 1, name='Mconv1_stage5_L3')
        #      .conv(7, 7, 128, 1, 1, name='Mconv2_stage5_L3')
        #      .conv(7, 7, 128, 1, 1, name='Mconv3_stage5_L3')
        #      .conv(7, 7, 128, 1, 1, name='Mconv4_stage5_L3')
        #      .conv(7, 7, 128, 1, 1, name='Mconv5_stage5_L3')
        #      .conv(1, 1, 128, 1, 1, name='Mconv6_stage5_L3')
        #      .conv(1, 1, output_channels*2, 1, 1, relu=False, name='Mconv7_stage5_L3'))

        (self.feed('Mconv7_stage5_L1',
                   'Mconv7_stage5_L2',
        #           'Mconv7_stage5_L3',
                   'concat_features')
             .concat(3, name='concat_stage6')
             .conv(7, 7, 128, 1, 1, name='Mconv1_stage6_L1')
             .conv(7, 7, 128, 1, 1, name='Mconv2_stage6_L1')
             .conv(7, 7, 128, 1, 1, name='Mconv3_stage6_L1')
             .conv(7, 7, 128, 1, 1, name='Mconv4_stage6_L1')
             .conv(7, 7, 128, 1, 1, name='Mconv5_stage6_L1')
             .conv(1, 1, 128, 1, 1, name='Mconv6_stage6_L1')
             .conv(1, 1, output_channels*2, 1, 1, relu=False, name='Mconv7_stage6_L1'))
             #.upsample(8, name='Mconv7_stage6_L1'))
             #.deconv2(output_channels*2, 2, name='upsample1_stage6_L1')
             #.deconv2(output_channels*2, 2, name='upsample2_stage6_L1')
             #.deconv2(output_channels*2, 2, name='Mconv7_stage6_L1'))

        (self.feed('concat_stage6')
             .conv(7, 7, 128, 1, 1, name='Mconv1_stage6_L2')
             .conv(7, 7, 128, 1, 1, name='Mconv2_stage6_L2')
             .conv(7, 7, 128, 1, 1, name='Mconv3_stage6_L2')
             .conv(7, 7, 128, 1, 1, name='Mconv4_stage6_L2')
             .conv(7, 7, 128, 1, 1, name='Mconv5_stage6_L2')
             .conv(1, 1, 128, 1, 1, name='Mconv6_stage6_L2')
             .conv(1, 1, output_channels, 1, 1, relu=False, name='Mconv7_stage6_L2'))
             #.upsample(8, name='Mconv7_stage6_L2'))
             #.deconv2(output_channels, 2, name='upsample1_stage6_L2')
             #.deconv2(output_channels, 2, name='upsample2_stage6_L2')
             #.deconv2(output_channels, 2, name='Mconv7_stage6_L2'))
        #(self.feed('concat_stage6')
        #     .conv(7, 7, 128, 1, 1, name='Mconv1_stage6_L3')
        #     .conv(7, 7, 128, 1, 1, name='Mconv2_stage6_L3')
        #     .conv(7, 7, 128, 1, 1, name='Mconv3_stage6_L3')
        #     .conv(7, 7, 128, 1, 1, name='Mconv4_stage6_L3')
        #     .conv(7, 7, 128, 1, 1, name='Mconv5_stage6_L3')
        #     .conv(1, 1, 128, 1, 1, name='Mconv6_stage6_L3')
        #     .conv(1, 1, output_channels*2, 1, 1, relu=False, name='Mconv7_stage6_L3'))

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
