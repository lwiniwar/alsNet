from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError
from sklearn.metrics import confusion_matrix
import tensorflow as tf
import numpy as np
import random

from dataset import Dataset
from alsNetHistory import AlsNetHistory

import os
import sys
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))

import tf_util
from pointnet_util import pointnet_sa_module, pointnet_fp_module


def simple_loss(labels, logits):
    return tf.losses.sparse_softmax_cross_entropy(labels, logits, scope='loss')


def fp_high_loss(labels, logits, factor=10):
    weights = tf.where(tf.logical_and(labels != 2, tf.argmax(logits) == 2), factor, 1)
    classify_loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits, scope='loss', weights=weights)
    return classify_loss


class AlsNetContainer(BaseEstimator, ClassifierMixin):
    def __init__(self,
                 output_base,
                 num_classes,
                 num_feat,
                 num_points,
                 learning_rate=0.1,
                 dropout=0.5,
                 activation_fn=tf.nn.relu,
                 optimizer_cls=tf.train.AdamOptimizer,
                 loss_fn=simple_loss,
                 initalizer=None,
                 arch=None,
                 score_sample=1,
                 gpu_id = None):
        self.output_dir = output_base
        self.num_classes = num_classes
        self.num_feat = num_feat
        self.num_points = num_points
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.activation_fn = activation_fn
        self.optimizer_cls = optimizer_cls
        self.loss_fn = loss_fn
        self.initalizer = initalizer
        self.arch = arch
        self.score_sample=score_sample
        self.gpu_id = gpu_id
        self._session = None
        self._graph = None

        self._config = tf.ConfigProto()
        self._config.gpu_options.allow_growth = True
        self._config.allow_soft_placement = True
        self._config.log_device_placement = False

        self._train_points_seen = 0
        self.train_history = AlsNetHistory()
        self.eval_history = AlsNetHistory()

    def _build_graph(self):
        """
        Build the graph
        :return: Nothing
        """
        if self.gpu_id is not None:
            dev = '/device:GPU:%s' % self.gpu_id
            print("Using GPU%s" % self.gpu_id)
        else:
            dev = 'CPU'
            print("Using CPU")

        with tf.device(dev):
            # define placeholders
            # input points
            points_in = tf.placeholder(tf.float32, shape=(1, self.num_points, self.num_feat + 3), name='points_in')
            # reference labels
            labels_ref = tf.placeholder(tf.int64, shape=(1, self.num_points), name='labels_ref')
            # training flag
            is_training = tf.placeholder(tf.bool, shape=(), name='is_training')

            # create network
            logits = self._dnn(points_in, is_training)

            # get loss
            loss = self.loss_fn(labels_ref, logits)

            # create optimizer
            optimizer = self.optimizer_cls(learning_rate=self.learning_rate)

            # set operations
            train_op = optimizer.minimize(loss, name='train')
            softmax_op = tf.nn.softmax(logits, name='softmax')

            # initalize variables
            init_op = tf.global_variables_initializer()
            saver = tf.train.Saver()

        # Make important vars/ops availiable instance-wide
        self._points_in = points_in
        self._labels_ref = labels_ref
        self._is_training = is_training
        self._logits = logits
        self._loss = loss
        self._train_op = train_op
        self._softmax_op = softmax_op
        self._init_op = init_op
        self._saver = saver

    def _dnn(self, points_in, is_training):
        """
        Central definition of the deep neural network: creates the sa and fp layers
        and handles dropout.
        :param points_in: tensor (batch x num_points x (3+num_feat)). input points (x,y,z,attr...)
        :param is_training: bool.
        :return: last layer of net
        """

        with tf.variable_scope('dnn'):
            ln_xyz = [tf.slice(points_in, [0, 0, 0], [-1, -1, 3])]    # point coordinates
            ln_feat = [tf.slice(points_in, [0, 0, 3], [-1, -1, -1])]  # point attributes

            for depth, step_dict in enumerate(self.arch):  # set abstraction
                xyz, feat = self._pointnet_sa(step_dict,
                                              ln_xyz[depth], ln_feat[depth],
                                              is_training,
                                              'sa_layer_%d' % (depth + 1))
                ln_xyz.append(xyz)
                ln_feat.append(feat)

            for depth, step_dict in enumerate(reversed(self.arch)):  # feature propagation
                depth = len(self.arch) - depth
                feat = self._pointnet_fp(step_dict,
                                         ln_xyz[depth-1], ln_xyz[depth],
                                         ln_feat[depth-1], ln_feat[depth],
                                         is_training,
                                         'fp_layer_%d' % (depth - 1))
                ln_feat[depth - 1] = feat

            l0_feats = ln_feat[0]
            net = tf_util.conv1d(l0_feats, 128, 1, padding='VALID', bn=True,
                                 is_training=is_training, scope='fc1', bn_decay=None)
            net = tf_util.dropout(net, keep_prob=(1-self.dropout), is_training=is_training, scope='dp1')
            net = tf_util.conv1d(net, self.num_classes, 1, padding='VALID', activation_fn=None, scope='fc2', name='net')
            return net

    def _pointnet_sa(self, arch_dict, xyz, feat, is_training, scope=""):
        """
        PointNet Set Abstraction layer (Qi et al. 2017)
        :param arch_dict: dictionary describing the architecture of this layer
        :param xyz: Tensor (batch x num_points x 3). coordinate triplets
        :param feat: Tensor (batch x num_points x num_feat). features for each point
        :param scope: name for the layers
        :return: xyz and features of the superpoint layer
        """
        li_xyz, li_feats, li_indices = pointnet_sa_module(xyz, feat,
                                                          npoint=arch_dict['npoint'],
                                                          radius=arch_dict['radius'],
                                                          nsample=arch_dict['nsample'],
                                                          mlp=arch_dict['mlp'],
                                                          pooling=arch_dict['pooling'],
                                                          mlp2=arch_dict['mlp2'],
                                                          group_all=False,
                                                          is_training=is_training,
                                                          bn_decay=None,
                                                          scope=scope)
        return li_xyz, li_feats

    def _pointnet_fp(self, arch_dict, xyz_to, xyz_from, feat_to, feat_from, is_training, scope=""):
        """
        PointNet Feature Propagation layer (Qi et al. 2017)
        :param arch_dict: dictionary describing the architecture of this layer
        :param xyz_to: Tensor (batch x num_points x 3). coordinate triplets
        :param xyz_from: Tensor (batch x num_points x 3). coordinate triplets
        :param feat_to: Tensor (batch x num_points x num_feat). features for each point
        :param feat_from: Tensor (batch x num_points x num_feat). features for each point
        :param scope: name for the layers
        :return: features interpolated to the next layer
        """
        li_feats = pointnet_fp_module(xyz_to, xyz_from,
                                      feat_to, feat_from,
                                      arch_dict['reverse_mlp'],
                                      is_training,
                                      bn_decay=None,
                                      scope=scope)
        return li_feats

    def close_session(self):
        if self._session:
            self._session.close()


    def fit_file(self, filenames_in, new_session=True, **kwargs):
        if new_session or self._graph is None:
            self.create_graph()
        for filename in filenames_in:
            ds = Dataset(filename)
            self.fit_one_epoch(ds.points_and_features, ds.labels)
        return self

    def create_graph(self):
        self.close_session()
        self._train_points_seen = 0
        self._graph = tf.Graph()
        with self._graph.as_default():
            self._build_graph()
        self._session = tf.Session(graph=self._graph, config=self._config)
        with self._session.as_default() as sess:
            sess.run(self._init_op)

    def fit(self, datasets, new_session=True):
        if new_session or self._graph is None:
            self.create_graph()
        for ds in datasets:
            points_in_single_ds, labels_single_ds = ds.points_and_features, ds.labels
            self.fit_one_epoch(points_in_single_ds, labels_single_ds)
            ds.unload()

    def fit_one_epoch(self, points_in, labels):
        with self._session.as_default() as sess:
            points_in = np.expand_dims(points_in, 0)
            labels = np.expand_dims(labels, 0)
            train, loss, class_prob = sess.run([self._train_op, self._loss, self._softmax_op],
                                            feed_dict={self._points_in: points_in,
                                                       self._labels_ref: labels,
                                                       self._is_training: True})
            new_classes = np.argmax(class_prob, axis=2)
            cm = confusion_matrix(labels[0], new_classes[0], range(self.num_classes))
            self._train_points_seen += len(labels[0]) *1e-6
            self.train_history.add_history_step(cm, self._train_points_seen, loss)

    def predict_probability(self, points_in):
        if not self._session:
            raise NotFittedError("This %s instance is not fitted yet" % self.__class__.__name__)
        with self._session.as_default() as sess:
            points_in = np.expand_dims(points_in, 0)
            return self._softmax_op.eval(feed_dict={self._points_in: points_in,
                                                    self._is_training: False})

    def predict_one_epoch(self, points_in):
        class_indices = np.argmax(self.predict_probability(points_in), axis=2)
        return class_indices

    def predict(self, points_in_mult):
        results = []
        for points_in in points_in_mult:
            pred_res = self.predict_one_epoch(points_in)
            results.append(pred_res[0])
        return results

    def _get_model_params(self):
        """Get all variable values (used for early stopping, faster than saving to disk)"""
        with self._graph.as_default():
            gvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        return {gvar.op.name: value for gvar, value in zip(gvars, self._session.run(gvars))}

    def _restore_model_params(self, model_params):
        """Set all variables to the given values (for early stopping, faster than loading from disk)"""
        gvar_names = list(model_params.keys())
        assign_ops = {gvar_name: self._graph.get_operation_by_name(gvar_name + "/Assign")
                      for gvar_name in gvar_names}
        init_values = {gvar_name: assign_op.inputs[1] for gvar_name, assign_op in assign_ops.items()}
        feed_dict = {init_values[gvar_name]: model_params[gvar_name] for gvar_name in gvar_names}
        self._session.run(assign_ops, feed_dict=feed_dict)

    def test_single(self, file_in, save_to=None, save_prob=False, unload=True):
        if isinstance(file_in, Dataset):
            ds = file_in
        else:
            ds = Dataset(file_in)
        probs = self.predict_probability(ds.points_and_features)
        new_classes = np.argmax(probs, axis=2)
        if save_to:
            Dataset.Save(save_to, ds.points_and_features,
                         ds.names, ds.labels, new_classes[0],
                         probs[0] if save_prob else None)

        cm = confusion_matrix(ds.labels, new_classes[0], range(self.num_classes))
        self.eval_history.add_history_step(cm, self._train_points_seen, 0)
        if unload: ds.unload()
        return np.count_nonzero(ds.labels == new_classes[0]) / self.num_points

    def save_model(self, path):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        self._saver.save(self._session, path)

    def load_model(self, path):
        if self._graph is None or self._session is None:
            self.close_session()
            self._train_points_seen = 0
            self._graph = tf.Graph()
            with self._graph.as_default():
                self._build_graph()
            self._session = tf.Session(graph=self._graph, config=self._config)

        with self._session.as_default() as sess:
            self._saver.restore(sess, path)

    def score(self, ds, sample_weight=None):
        from sklearn.metrics import accuracy_score
        try:
            samples = random.sample(ds, self.score_sample)
        except ValueError:  # too few samples --> take whatever we have
            samples = ds
        scores = []
        for sample in samples:
            X = sample.points_and_features
            y = sample.labels
            score = accuracy_score(np.array(y), np.array(self.predict_one_epoch(X)[0]), sample_weight=sample_weight)
            print("Current Accuracy score: %s" % score)
            scores.append(score)
            sample.unload()
        return np.mean(scores)
