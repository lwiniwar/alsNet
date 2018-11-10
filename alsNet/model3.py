import os
import sys
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tensorflow as tf
import numpy as np
import tf_util
from pointnet_util import pointnet_sa_module, pointnet_fp_module

arch = [
    {
        'npoint': 4*4096,
        'radius': 1,
        'nsample': 32,
        'mlp': [128, 128, 128],
        'pooling': 'max',
        'mlp2': None,
        'reverse_mlp': [128,128]
    },
    {
        'npoint': 2*4096,
        'radius': 5,
        'nsample': 64,
        'mlp': [128, 128, 256],
        'pooling': 'max',
        'mlp2': None,
        'reverse_mlp': [256,256]
    },
    {
        'npoint': 1*4096,
        'radius': 15,
        'nsample': 64,
        'mlp': [128, 128, 256],
        'pooling': 'max',
        'mlp2': None,
        'reverse_mlp': [256,256]
    },
    #{
    #    'npoint': 16,
    #    'radius': 999,
    #    'nsample': 4096,
    #    'mlp': [256, 256, 512],
    #    'pooling': 'max_and_avg',
    #    'mlp2': None,
    #    'reverse_mlp': [512,512]
    #}
]


def placeholder_inputs(batch_size, points_in, num_feat):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, points_in, num_feat), name='pointcloud_in')
    labels_pl = tf.placeholder(tf.int64, shape=(batch_size, points_in), name='labels')
    #smpws_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point))
    return pointclouds_pl, labels_pl


def get_model(point_cloud, is_training, num_class, dropout, bn_decay=None):
    """ Semantic segmentation PointNet, input is BxNx3, output Bxnum_class """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}
    l0_xyz = tf.slice(point_cloud, [0,0,0], [-1,-1,3])  # point coordinates
    l0_points = tf.slice(point_cloud, [0,0,3], [-1,-1,-1])  #point attributes
    end_points['l0_xyz'] = l0_xyz

    ln_xyz = [l0_xyz]
    ln_points = [l0_points]
    ln_indices = [None]
    for depth, step_dict in enumerate(arch):
        li_xyz, li_points, li_indices = pointnet_sa_module(ln_xyz[depth], ln_points[depth],
                                                           npoint=step_dict['npoint'],
                                                           radius=step_dict['radius'],
                                                           nsample=step_dict['nsample'],
                                                           mlp=step_dict['mlp'],
                                                           pooling=step_dict['pooling'],
                                                           mlp2=step_dict['mlp2'],
                                                           group_all=False,
                                                           is_training=is_training,
                                                           bn_decay=bn_decay,
                                                           scope='layer%d' % (depth+1))
        ln_xyz.append(li_xyz)
        #end_points['l%d_xyz' % (depth+1)] = li_xyz  # debug subsampled points
        ln_points.append(li_points)
        ln_indices.append(li_indices)

    for depth, step_dict in enumerate(reversed(arch)):
        depth = len(arch) - depth
        li_points = pointnet_fp_module(ln_xyz[depth-1], ln_xyz[depth],
                                       ln_points[depth-1], ln_points[depth],
                                       step_dict['reverse_mlp'],
                                       is_training,
                                       bn_decay,
                                       scope='fa_layer%d' % (depth-1))
        ln_points[depth-1] = li_points

    l0_points = ln_points[0]

    # Set Abstraction layers
    #l1_xyz, l1_points, l1_indices = pointnet_sa_module(l0_xyz, l0_points, npoint=2048, radius=3, nsample=64, mlp=[64,64,128], pooling='max_and_avg', mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer1')
    #l2_xyz, l2_points, l2_indices = pointnet_sa_module(l1_xyz, l1_points, npoint=1024, radius=10, nsample=32, mlp=[128,128,128], pooling='max_and_avg', mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer2')
    #l3_xyz, l3_points, l3_indices = pointnet_sa_module(l2_xyz, l2_points, npoint=512, radius=15, nsample=32, mlp=[128,128,256], pooling='max_and_avg', mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer3')
    #l4_xyz, l4_points, l4_indices = pointnet_sa_module(l3_xyz, l3_points, npoint=256, radius=50, nsample=32, mlp=[256,256,512], pooling='max_and_avg', mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer4')
    #l5_xyz, l5_points, l5_indices = pointnet_sa_module(l4_xyz, l4_points, npoint=100, radius=25, nsample=32, mlp=[512,512,1024], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer5')
    # debug line:
    # l4_points = tf.Print(l1_points, [l0_xyz, l0_points, l1_xyz, l1_points], 'ln-points', -1, 12)
    #end_points['l1_xyz'] = l1_xyz
    #end_points['l2_xyz'] = l2_xyz
    #end_points['l3_xyz'] = l3_xyz
    #end_points['l4_xyz'] = l4_xyz
    #end_points['l5_xyz'] = l5_xyz

    # Feature Propagation layers
    #l4_points = pointnet_fp_module(l4_xyz, l5_xyz, l4_points, l5_points, [512,512], is_training, bn_decay, scope='fa_layer0')
    #l3_points = pointnet_fp_module(l3_xyz, l4_xyz, l3_points, l4_points, [256,256], is_training, bn_decay, scope='fa_layer1')
    #l2_points = pointnet_fp_module(l2_xyz, l3_xyz, l2_points, l3_points, [256,256], is_training, bn_decay, scope='fa_layer2')
    #l1_points = pointnet_fp_module(l1_xyz, l2_xyz, l1_points, l2_points, [128,128], is_training, bn_decay, scope='fa_layer3')
    #l0_points = pointnet_fp_module(l0_xyz, l1_xyz, l0_points, l1_points, [128,128,128], is_training, bn_decay, scope='fa_layer4')

    # FC layers
    net = tf_util.conv1d(l0_points, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)
    end_points['feats'] = net
    net = tf_util.dropout(net, keep_prob=1-dropout, is_training=is_training, scope='dp1')
    net = tf_util.conv1d(net, num_class, 1, padding='VALID', activation_fn=None, scope='fc2', name='net')

    return net, end_points


def get_loss(pred, label):
    """ pred: BxNxC,
        label: BxN,
        smpw: BxN """
    #classify_loss = tf.losses.sparse_softmax_cross_entropy(labels=label, logits=pred, scope='loss')
    weights = tf.where(tf.logical_and(label != 2, tf.argmax(pred) == 2), 10, 1)
    classify_loss = tf.losses.sparse_softmax_cross_entropy(labels=label, logits=pred, scope='loss', weights=weights)
    #classify_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=pred, name='loss')
    # false negatives are less of a problem than false positives
    # pred_val = np.argmax(pred)
    # fn = np.count_nonzero(np.logical_and(label == 2, pred_val != 2))
    # fp = np.count_nonzero(np.logical_and(label != 2, pred_val == 2))
    # tn = np.count_nonzero(np.logical_and(label != 2, pred_val != 2))
    # tp = np.count_nonzero(np.logical_and(label == 2, pred_val == 2))
    # precision = tp / (tp+fp+0.01)  # correctness
    # recall = tp / (tp+fn+0.01)  # completeness
    # f1_score = 2*precision*recall / (precision+recall+0.01)
    # classify_loss = 1-f1_score
    tf.summary.scalar('classify_loss', classify_loss)
    return classify_loss
