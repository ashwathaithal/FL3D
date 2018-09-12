'''
 Peng YUN Created on Wed Jun 20 2018

 Copyright (c) 2018 RAM-LAB
'''

from model.layers import *
import tensorflow as tf
class FCN3D(object):

    def __init__(self, sess, scale=8, voxel_shape=(300, 300, 300), is_training=True, alpha=1, beta=1.5, eta=1, gamma=2):
        self.voxel = tf.placeholder(tf.float32, [None, voxel_shape[0], voxel_shape[1], voxel_shape[2], 1], name="voxel")
        self.is_training = tf.placeholder(tf.bool, name='phase_train') if is_training else None
        self.global_step = tf.Variable(1, trainable=False, name='global_step')
        self.epoch = tf.Variable(0, trainable=False, name='epoch')
        self.epoch_add_op = self.epoch.assign(self.epoch + 1)
        self.scale = scale
        with tf.variable_scope("FCN3D") as scope:
            self.layer1 = conv3d_layer(self.voxel, 32, [5,5,5], [2,2,2], use_bias=False, activation=tf.nn.relu, name="CONV3d-1")
            self.layer1_bn = batch_norm(self.layer1, self.is_training, name="BN-1") 
            self.layer2 = conv3d_layer(self.layer1_bn, 64, [5,5,5], [2,2,2], use_bias=False, activation=tf.nn.relu, name="CONV3d-2")
            self.layer2_bn = batch_norm(self.layer2, self.is_training, name="BN-2")
            self.layer3 = conv3d_layer(self.layer2_bn, 96, [3,3,3], [2,2,2], use_bias=False, activation=tf.nn.relu, name="CONV3d-3")
            self.layer3_bn = batch_norm(self.layer3, self.is_training, name="BN-3")
            self.layer4 = conv3d_layer(self.layer3_bn, 96, [3,3,3], [1,1,1], use_bias=False, activation=tf.nn.relu, name="CONV3d-4")
            self.layer4_bn = batch_norm(self.layer4, self.is_training, name="BN-4")
            if self.scale == 8:
                self.objectness = conv3d_layer(self.layer4_bn, 1, [3,3,3], [1,1,1], use_bias=False, activation=None, name="CONV3d-Obj")
                self.cordinate = conv3d_layer(self.layer4_bn, 24, [3,3,3], [1,1,1], use_bias=False, activation=None, name="CONV3d-Cor")
                self.y = tf.sigmoid(self.objectness)
            elif self.scale == 4:
                self.objectness = conv3d_transpose_layer(self.layer4_bn, 1, [3,3,3], [2,2,2], use_bias=False, activation=None, name="DeCONV3d-Obj")
                self.cordinate = conv3d_transpose_layer(self.layer4_bn, 24, [3,3,3], [2,2,2], use_bias=False, activation=None, name="DeCONV3d-Cor")
                self.y = tf.sigmoid(self.objectness)  
            elif self.scale == 2:
                self.layer5 = conv3d_transpose_layer(self.layer4_bn, 64, [3,3,3], [2,2,2], use_bias=False, activation=tf.nn.relu, name="DeCONV3d-5")
                self.layer5_bn = batch_norm(self.layer5, self.is_training, name="BN-5")
                self.objectness = conv3d_transpose_layer(self.layer5_bn, 1, [3,3,3], [2,2,2], use_bias=False, activation=None, name="DeCONV3d-Obj")
                self.cordinate = conv3d_transpose_layer(self.layer5_bn, 24, [3,3,3], [2,2,2], use_bias=False, activation=None, name="DeCONV3d-Cor")
                self.y = tf.sigmoid(self.objectness)                                   
            else:
                raise NotImplementedError

        self.g_map = tf.placeholder(tf.float32, self.cordinate.get_shape().as_list()[:4], name='g_map')
        self.g_cord = tf.placeholder(tf.float32, self.cordinate.get_shape().as_list(), name='g_cord')
        self.is_obj_loss = None
        self.non_obj_loss = None
        self.obj_loss = None
        self.cord_loss = None
        self.loss = None
        self.optimizer = None
        self.vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        self.params = tf.trainable_variables()
        self.set_loss(alpha=alpha, beta=beta, eta=eta, gamma=gamma)
        self.box2d_ind_after_nms = None
        self.boxes2d = tf.placeholder(tf.float32, [None, 4], name='boxes2d')
        self.boxes2d_scores = tf.placeholder(tf.float32, [None], name='boxes2d_scores')

        if is_training:
            initialized_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="FCN3D")
            sess.run(tf.variables_initializer(initialized_var))

        self.train_summary = tf.summary.merge([
            tf.summary.scalar('train/loss', self.loss),
            tf.summary.scalar('train/cord_loss', self.cord_loss),
            tf.summary.scalar('train/obj_loss', self.obj_loss),
            tf.summary.scalar('train/is_obj_loss', self.is_obj_loss),
            tf.summary.scalar('train/non_obj_loss', self.non_obj_loss),
            *[tf.summary.histogram(each.name, each) for each in self.vars + self.params]
        ])

        self.validate_summary = tf.summary.merge([
            tf.summary.scalar('validate/loss', self.loss),
            tf.summary.scalar('validate/cord_loss', self.cord_loss),
            tf.summary.scalar('validate/obj_loss', self.obj_loss),
            tf.summary.scalar('validate/is_obj_loss', self.is_obj_loss),
            tf.summary.scalar('validate/non_obj_loss', self.non_obj_loss)
        ])

    def set_loss(self, alpha=1, beta=1.5, eta=1, gamma=2):
        # # Classification
        g_map_sum = tf.reduce_sum(self.g_map, [0, 1, 2, 3])
        g_map_sum = tf.cast(g_map_sum, tf.int32)
        non_gmap_sum = tf.size(self.g_map) - g_map_sum
        g_map_sum = tf.cast(g_map_sum, tf.float32)
        non_gmap_sum = tf.cast(non_gmap_sum, tf.float32)

        non_gmap = tf.subtract(tf.ones_like(self.g_map, dtype=tf.float32), self.g_map)
        elosion = 1e-6
        if gamma == 0:
            print("set_loss: Normal Loss")
        elif gamma > 0:
            print("set_loss: Focal Loss, Gamma is {}".format(gamma))
        else:
            return NotImplementedError
        self.is_obj_loss  = -tf.reduce_sum(tf.multiply(self.g_map * (1 -  self.y[:, :, :, :, 0] + elosion) ** gamma, 
                                                       tf.log(    self.y[:, :, :, :, 0] + elosion))) / (g_map_sum+elosion) * alpha
        self.non_obj_loss = -tf.reduce_sum(tf.multiply(  non_gmap * (     self.y[:, :, :, :, 0] + elosion) ** gamma, 
                                                       tf.log(1 - self.y[:, :, :, :, 0] + elosion))) / (non_gmap_sum+elosion) * beta            
        self.obj_loss =  tf.add(self.is_obj_loss, self.non_obj_loss) * eta
        
        # Regression
        cord_diff = tf.multiply(self.g_map, tf.reduce_sum(tf.square(tf.subtract(self.cordinate, self.g_cord)), 4))
        self.cord_loss = tf.multiply(tf.reduce_sum(cord_diff), 0.02)
        self.loss = self.obj_loss + self.cord_loss

    def set_optimizer(self, lr=0.001):
        with tf.variable_scope("optmizer"):
            opt = tf.train.AdamOptimizer(lr)
            self.optimizer = opt.minimize(self.loss, global_step=self.global_step)
    
    def set_nms(self, max_output_size, iou_threshold):
        self.box2d_ind_after_nms = tf.image.non_max_suppression(
                self.boxes2d, self.boxes2d_scores, max_output_size=max_output_size, iou_threshold=iou_threshold)

    def train(self, sess, batch_x, batch_g_map, batch_g_cord):
        _, reg_loss, obj_loss, pos_cls_loss, neg_cls_loss, loss, summary = sess.run([self.optimizer, self.cord_loss, self.obj_loss, self.is_obj_loss, self.non_obj_loss, self.loss, self.train_summary], 
                                                            feed_dict={self.voxel: batch_x, 
                                                            self.g_map: batch_g_map, 
                                                            self.g_cord: batch_g_cord, 
                                                            self.is_training:True})    
        return pos_cls_loss, neg_cls_loss, obj_loss, reg_loss, loss, summary

    def validate(self, sess, batch_x, batch_g_map, batch_g_cord):
        reg_loss, obj_loss, pos_cls_loss, neg_cls_loss, loss, summary = sess.run([self.cord_loss, self.obj_loss, self.is_obj_loss, self.non_obj_loss, self.loss, self.validate_summary], 
                                                            feed_dict={self.voxel: batch_x, 
                                                            self.g_map: batch_g_map, 
                                                            self.g_cord: batch_g_cord, 
                                                            self.is_training:False})  
        # HERE set self.is_training as False does not make sense                                                                                                                        
        return pos_cls_loss, neg_cls_loss, obj_loss, reg_loss, loss, summary          

    def evaluate(self, sess, voxel_x):              
        objectness, cordinate, y_pred = sess.run([self.objectness, self.cordinate, self.y], 
                                                            feed_dict={self.voxel: voxel_x})
        objectness = objectness[0, :, :, :, 0]
        cordinate = cordinate[0]
        y_pred = y_pred[0, :, :, :, 0]

        return objectness, cordinate, y_pred

class DenseNet(object):
    def __init__(self, sess, scale, voxel_shape=(300, 300, 300), is_training=True, alpha=1, beta=1.5, eta=1):
        self.voxel = tf.placeholder(tf.float32, [None, voxel_shape[0], voxel_shape[1], voxel_shape[2], 1])
        self.phase_train = tf.placeholder(tf.bool, name='phase_train') if is_training else None
        self.global_step = tf.Variable(1, trainable=False)
        self.epoch = tf.Variable(0, trainable=False)
        self.epoch_add_op = self.epoch.assign(self.epoch + 1)
        self.scale = scale
        # self.filters = filters
        self.is_training = is_training

        with tf.variable_scope("DenseNet3D") as scope:
            self.conv3d_1 = conv3d_layer(self.voxel, filter=16, kernel=[7,7,7], stride=[1,1,1], use_bias=False, activation=None, name="CONV3d-1")
            self.dense_1 = self.dense_block(input_x=self.conv3d_1, nb_layers=6, nb_blocks=2, layer_name='dense_1')
            self.trans_1 = self.transition_layer(self.dense_1, filters=6, scope='trans_1')
            self.dense_2 = self.dense_block(input_x=self.trans_1, nb_layers=12, nb_blocks=2, layer_name='dense_2')
            self.trans_2 = self.transition_layer(self.dense_2, filters=12, scope='trans_2')
            self.dense_3 = self.dense_block(input_x=self.trans_2, nb_layers=48, nb_blocks=2, layer_name='dense_3')
            self.trans_3 = self.transition_layer(self.dense_3, filters=48, scope='trans_3')
            self.dense_final = self.dense_block(input_x=self.trans_3, nb_layers=32, nb_blocks=2, layer_name='dense_final')
            self.linear_batch = batch_norm(self.dense_final, phase_train=self.is_training, name='linear_batch')
            self.linear_batch = tf.nn.relu(self.linear_batch)

            if self.scale == 4:
                self.objectness = conv3d_transpose_layer(self.linear_batch, filter=2, kernel=[3,3,3], stride=[2,2,2], use_bias=False, activation=None, name="DECONV3d-Obj")
                self.cordinate = conv3d_transpose_layer(self.linear_batch, filter=24, kernel=[3,3,3], stride=[2,2,2], use_bias=False, activation=None, name="DECONV3d-Cor")
                self.y = tf.nn.softmax(self.objectness, dim=-1)   
            elif self.scale == 2:
                self.conv3d_trans = conv3d_transpose_layer(self.linear_batch, filter=32, kernel=[3,3,3], stride=[2,2,2], use_bias=False, activation=None, name="DECONV3d-1")
                self.objectness = conv3d_transpose_layer(self.conv3d_trans, filter=2, kernel=[3,3,3], stride=[2,2,2], use_bias=False, activation=None, name="DECONV3d-Obj")
                self.cordinate = conv3d_transpose_layer(self.conv3d_trans, filter=24, kernel=[3,3,3], stride=[2,2,2], use_bias=False, activation=None, name="DECONV3d-Cor")                
                self.y = tf.nn.softmax(self.objectness, dim=-1)   
            else :
                raise NotImplementedError

        self.g_map = tf.placeholder(tf.float32, self.cordinate.get_shape().as_list()[:4])
        self.g_cord = tf.placeholder(tf.float32, self.cordinate.get_shape().as_list())
        self.is_obj_loss = None
        self.non_obj_loss = None
        self.obj_loss = None
        self.cord_loss = None
        self.loss = None
        self.optimizer = None
        self.box2d_ind_after_nms = None
        self.vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        self.params = tf.trainable_variables()
        self.set_loss(alpha=alpha, beta=beta, eta=eta)
        self.boxes2d = tf.placeholder(tf.float32, [None, 4])
        self.boxes2d_scores = tf.placeholder(tf.float32, [None])
        
        if self.is_training:
            initialized_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="DenseNet3D")
            sess.run(tf.variables_initializer(initialized_var))

        self.train_summary = tf.summary.merge([
            tf.summary.scalar('train/loss', self.loss),
            tf.summary.scalar('train/cord_loss', self.cord_loss),
            tf.summary.scalar('train/obj_loss', self.obj_loss),
            tf.summary.scalar('train/is_obj_loss', self.is_obj_loss),
            tf.summary.scalar('train/non_obj_loss', self.non_obj_loss),
            *[tf.summary.histogram(each.name, each) for each in self.vars + self.params]
        ])

        self.validate_summary = tf.summary.merge([
            tf.summary.scalar('validate/loss', self.loss),
            tf.summary.scalar('validate/cord_loss', self.cord_loss),
            tf.summary.scalar('validate/obj_loss', self.obj_loss),
            tf.summary.scalar('validate/is_obj_loss', self.is_obj_loss),
            tf.summary.scalar('validate/non_obj_loss', self.non_obj_loss)
        ])

    def bottleneck_layer(self, x, filters, scope):
        with tf.name_scope(scope):
            x = batch_norm(inputs=x,phase_train=self.is_training,name=scope+'_batch1')
            x = tf.nn.relu(x)
            x = conv3d_layer(x, filter=4*filters, kernel=[1,1,1], stride=[1,1,1], use_bias=False, activation=None, name=scope+"_conv1")
            x = tf.layers.dropout(inputs=x,rate=0.4,training=self.is_training)
            x = batch_norm(inputs=x,phase_train=self.is_training,name=scope+'_batch2')
            x = tf.nn.relu(x)
            x = conv3d_layer(x, filter=filters, kernel=[3,3,3], stride=[1,1,1], use_bias=False, activation=None, name=scope+"_conv2")
            x = tf.layers.dropout(inputs=x,rate=0.4,training=self.is_training)
            return x
    
    def transition_layer(self,x, filters, scope):
        with tf.name_scope(scope):
            x = batch_norm(inputs=x,phase_train=self.is_training,name=scope+'_batch1')
            x = tf.nn.relu(x)
            x = conv3d_layer(x, filter=filters, kernel=[1,1,1], stride=[1,1,1], use_bias=False, activation=None, name=scope+"_conv1")
            x = tf.layers.dropout(inputs=x,rate=0.4,training=self.is_training)
            x = average_pool(x,[1,2,2,2,1],[1,2,2,2,1],name=scope+"avg3d")
            return x
            
    def dense_block(self,input_x,nb_layers,nb_blocks, layer_name):
        with tf.name_scope(layer_name):
            layers_concat =list()
            layers_concat.append(input_x)
            x =self.bottleneck_layer(input_x, nb_layers, scope =layer_name+'_bottle_' + str(0))
            layers_concat.append(x)
            for i in range(nb_blocks-1):
                x = tf.concat(layers_concat,axis=4)
                x = self.bottleneck_layer(x, nb_layers, scope =layer_name + '_bottle_'+ str(i+1))
                layers_concat.append(x)
            x = tf.concat(layers_concat,axis=4)
            return x

    def set_loss(self, alpha=1, beta=1.5, eta=1):
        # # Classification
        # g_map_sum = tf.reduce_sum(self.g_map, [0, 1, 2, 3])
        # g_map_sum = tf.cast(g_map_sum, tf.int32)
        # non_gmap_sum = tf.size(self.g_map) - g_map_sum
        # g_map_sum = tf.cast(g_map_sum, tf.float32)
        # non_gmap_sum = tf.cast(non_gmap_sum, tf.float32)

        # non_gmap = tf.subtract(tf.ones_like(self.g_map, dtype=tf.float32), self.g_map)
        # elosion = 1e-6
        # self.is_obj_loss  = -tf.reduce_sum(tf.multiply(self.g_map, tf.log(self.y[:, :, :, :, 0] + elosion))) / g_map_sum * alpha
        # self.non_obj_loss = -tf.reduce_sum(tf.multiply(  non_gmap, tf.log(self.y[:, :, :, :, 1] + elosion))) / non_gmap_sum * beta
        # self.obj_loss =  tf.add(self.is_obj_loss, self.non_obj_loss) * eta

        # # Regression
        # cord_diff = tf.multiply(self.g_map, tf.reduce_sum(tf.square(tf.subtract(self.cordinate, self.g_cord)), 4))
        # self.cord_loss = tf.multiply(tf.reduce_sum(cord_diff), 0.02)
        # self.loss = self.obj_loss + self.cord_loss

        non_gmap = tf.subtract(tf.ones_like(self.g_map, dtype=tf.float32), self.g_map)
        elosion = 0.00001
        y = self.y
        self.is_obj_loss = -tf.reduce_sum(tf.multiply(self.g_map,  tf.log(y[:, :, :, :, 0] + elosion)))
        self.non_obj_loss = tf.multiply(-tf.reduce_sum(tf.multiply(non_gmap, tf.log(y[:, :, :, :, 1] + elosion))), 0.0008)
        cross_entropy = tf.add(self.is_obj_loss, self.non_obj_loss)
        self.obj_loss = cross_entropy

        cord_diff = tf.multiply(self.g_map, tf.reduce_sum(tf.square(tf.subtract(self.cordinate, self.g_cord)), 4))
        self.cord_loss = tf.multiply(tf.reduce_sum(cord_diff), 0.02)
        self.loss = tf.add(self.obj_loss, self.cord_loss)
        
    def set_optimizer(self, lr=0.001):
        opt = tf.train.AdamOptimizer(lr)
        self.optimizer = opt.minimize(self.loss, global_step=self.global_step)

    def set_nms(self, max_output_size, iou_threshold):
        self.box2d_ind_after_nms = tf.image.non_max_suppression(
                self.boxes2d, self.boxes2d_scores, max_output_size=max_output_size, iou_threshold=iou_threshold)

    def train(self, sess, batch_x, batch_g_map, batch_g_cord):
        _, reg_loss, obj_loss, pos_cls_loss, neg_cls_loss, loss, summary = sess.run([self.optimizer, self.cord_loss, self.obj_loss, self.is_obj_loss, self.non_obj_loss, self.loss, self.train_summary], 
                                                            feed_dict={self.voxel: batch_x, 
                                                            self.g_map: batch_g_map, 
                                                            self.g_cord: batch_g_cord, 
                                                            self.phase_train:True})    
        return pos_cls_loss, neg_cls_loss, obj_loss, reg_loss, loss, summary

    def validate(self, sess, batch_x, batch_g_map, batch_g_cord):
        reg_loss, obj_loss, pos_cls_loss, neg_cls_loss, loss, summary = sess.run([self.cord_loss, self.obj_loss, self.is_obj_loss, self.non_obj_loss, self.loss, self.validate_summary], 
                                                            feed_dict={self.voxel: batch_x, 
                                                            self.g_map: batch_g_map, 
                                                            self.g_cord: batch_g_cord, 
                                                            self.phase_train:True})      
        return pos_cls_loss, neg_cls_loss, obj_loss, reg_loss, loss, summary        

    def evaluate(self, sess, voxel_x):              
        objectness, cordinate, y_pred = sess.run([self.objectness, self.cordinate, self.y], feed_dict={self.voxel: voxel_x})
        objectness = objectness[0, :, :, :, 0]
        cordinate = cordinate[0]
        y_pred = y_pred[0, :, :, :, 0]

        return objectness, cordinate, y_pred
