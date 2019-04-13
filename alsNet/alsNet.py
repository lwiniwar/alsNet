import argparse
import sys
import logging
import os
import glob
import random
import tensorflow as tf
import numpy as np


from dataset import Dataset
#import model
import model3 as model

def exp_learning_rate(learning_rate, global_step, step, decay, staircase=False):
    learning_rate = tf.train.exponential_decay(
                        learning_rate,  # Base learning rate.
                        global_step,  # Current index into the dataset.
                        step,          # Decay step.
                        decay,          # Decay rate.
                        staircase=False)
    learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
    return learning_rate


class AlsNetContainer:
    def __init__(self, logDir, learning_rate=0.01, dropout=0.5, logger=None, dev='/device:GPU:1'):
        self.logDir = logDir
        if not os.path.exists(self.logDir):
            os.makedirs(self.logDir)
        self.pl = {}  # placeholders
        self.op = {}  # operations
        self.cumsum_train = 0
        self.cumsum_train_correct = 0
        self.cumsum_test = 0
        self.cumsum_test_correct = 0
        self.graph = tf.Graph()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = False
        config.log_device_placement = True
        self.sess = tf.Session(config=config, graph=self.graph)
        self.logger = logger
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.dev = dev

    def prepare(self, num_classes, num_feat, points_in, batch_size=1):
        logging.info("""Model
  __  __   ___   ___   ___  _    
 |  \/  | / _ \ |   \ | __|| |   
 | |\/| || (_) || |) || _| | |__ 
 |_|  |_| \___/ |___/ |___||____|
 """)
        logging.info("Number of features (without xyz): %d" % num_feat)
        logging.info("Number of classes (output size): %d" % num_classes)
        logging.info("Number of points per batch: %d" % points_in)
        logging.info("Number batches per step: %d" % batch_size)
        #print("Number of points in Dataset: %d" % NUM_POINT)
        self.num_classes = num_classes
        if self.logger and hasattr(model, 'arch'):
            self.logger.arch = model.arch
        with self.graph.as_default():
            with tf.device(self.dev):
                self.pl['pointclouds_pl'], self.pl['labels_pl'] = model.placeholder_inputs(batch_size, points_in, num_feat + 3)  # plus xyz
                self.pl['is_training_pl'] = tf.placeholder(tf.bool, shape=(), name='is_training')
                self.op['pred'], self.op['end_points'] = model.get_model(self.pl['pointclouds_pl'],
                                                                         self.pl['is_training_pl'],
                                                                         num_classes,
                                                                         self.dropout)
                self.op['loss'] = model.get_loss(self.op['pred'], self.pl['labels_pl'])
                tf.summary.scalar('loss', self.op['loss'])

                #correct = tf.equal(tf.argmax(self.op['pred'], 2), tf.to_int64(self.pl['labels_pl']))
                #accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE * NUM_POINT)
                #tf.summary.scalar('accuracy', accuracy)
                global_step = tf.Variable(0, trainable=False)
                self.increment_global_step = tf.assign(global_step, global_step + 1)
                step_rate = 10
                decay = 0.95
                self.op['learning_rate'] = exp_learning_rate(self.learning_rate, global_step, step_rate, decay, staircase=False)

                optimizer = tf.train.AdamOptimizer(self.op['learning_rate'])
                self.op['train_op'] = optimizer.minimize(self.op['loss'], name='train')

                self.op['softmax'] = tf.nn.softmax(self.op['pred'], name='softmax')

                #saver = tf.train.Saver()

            #self.op['merged'] = tf.summary.merge_all()
            #self.train_writer = tf.summary.FileWriter(os.path.join(self.logDir, 'train'), self.sess.graph)
            #self.test_writer = tf.summary.FileWriter(os.path.join(self.logDir, 'test'), self.sess.graph)

            # Init variables
            init = tf.global_variables_initializer()
            self.sess.run(init)

    def train(self, train_ds):
        with self.graph.as_default():
            is_training = True
            feed_dict = {self.pl['pointclouds_pl']: np.expand_dims(train_ds.points_and_features, 0),
                         self.pl['labels_pl']: np.expand_dims(train_ds.labels, 0),
                         self.pl['is_training_pl']: is_training, }
            _, loss_val, pred_val = self.sess.run([self.op['train_op'],
                                                   self.op['loss'],
                                                   self.op['pred']],
                                                   feed_dict=feed_dict)
            pred_val = np.argmax(pred_val, 2)
            correct = np.sum(pred_val == train_ds.labels)
            self.cumsum_train_correct += correct
            self.cumsum_train += len(train_ds)
            logging.info("Cum. Accuracy: %.2f %%" % (100*self.cumsum_train_correct/self.cumsum_train))
            logging.info("Curr. Accuracy: %.2f %%" % (100*correct/len(train_ds)))
            logging.info("Curr. Loss: %.5f" % loss_val)

    def train_batch(self, points_and_features, labels):
        with self.graph.as_default():
            is_training = True
            feed_dict = {self.pl['pointclouds_pl']: points_and_features,
                         self.pl['labels_pl']: labels,
                         self.pl['is_training_pl']: is_training, }
            if np.count_nonzero(labels == 6)/(labels.shape[0] * labels.shape[1]) > 0.1:
                logging.debug("Building batch (> 10 %)")
            if (np.count_nonzero(labels > 2) - np.count_nonzero(labels > 6))/(labels.shape[0] * labels.shape[1]) > 0.4:
                logging.debug("Vegetation batch (> 40%)")
            if np.count_nonzero(labels == 2)/(labels.shape[0] * labels.shape[1]) > 0.9:
                logging.debug("Ground batch (> 90%)")

            _, loss_val, pred_val, end_points, lr, _ = self.sess.run([self.op['train_op'],
                                                       self.op['loss'],
                                                       self.op['pred'],
                                                       self.op['end_points'],
                                                       self.op['learning_rate'],
                                                       self.increment_global_step],
                                                       feed_dict=feed_dict)

            if False:  # debug: save superpoint-clouds
                if not os.path.exists(os.path.join(self.logDir, 'subsampl')):
                    os.makedirs(os.path.join(self.logDir, 'subsampl'))
                logging.debug("Saving subsampl to %s" % os.path.join(self.logDir, 'subsampl'))
                Dataset.Save(os.path.join(self.logDir, 'subsampl', 'l0.laz'), end_points['l0_xyz'][0])
                Dataset.Save(os.path.join(self.logDir, 'subsampl', 'l1.laz'), end_points['l1_xyz'][0])
                Dataset.Save(os.path.join(self.logDir, 'subsampl', 'l2.laz'), end_points['l2_xyz'][0])
                Dataset.Save(os.path.join(self.logDir, 'subsampl', 'l3.laz'), end_points['l3_xyz'][0])

            pred_val = np.argmax(pred_val, 2)
            correct = np.sum(pred_val == labels)
            self.cumsum_train_correct += correct
            points_in_batch = labels.shape[0] * labels.shape[1]
            self.cumsum_train += points_in_batch
            logging.info(" --- Cum. Accuracy: %.2f %%" % (100*self.cumsum_train_correct/self.cumsum_train))
            logging.info(" --- Curr. Accuracy: %.2f %%" % (100*correct/points_in_batch))
            logging.info(" --- Curr. avg. Loss: %.5f" % np.mean(loss_val))
            if self.logger:
                self.logger.points_seen.append(self.cumsum_train * 1e-6)
                self.logger.losses.append(np.mean(loss_val))
                self.logger.cumaccuracy_train.append(100*self.cumsum_train_correct/self.cumsum_train)
                self.logger.accuracy_train.append(100*correct/points_in_batch)
                self.logger.perc_building.append(100*np.count_nonzero(labels == 6)/points_in_batch)
                self.logger.perc_ground.append(100*np.count_nonzero(labels == 2)/points_in_batch)
                self.logger.perc_hi_veg.append(100*np.count_nonzero(labels == 5)/points_in_batch)
                self.logger.perc_med_veg.append(100*np.count_nonzero(labels == 4)/points_in_batch)
                self.logger.perc_lo_veg.append(100*np.count_nonzero(labels == 3)/points_in_batch)
                self.logger.perc_water.append(100*np.count_nonzero(labels == 9)/points_in_batch)
                self.logger.perc_rest.append(0)
                self.logger.lr.append(lr)
                self.logger.save()

    def train_chunk(self, points_and_features, labels):
        with self.graph.as_default():
            is_training = True
            feed_dict = {self.pl['pointclouds_pl']: np.expand_dims(points_and_features, 0),
                         self.pl['labels_pl']: np.expand_dims(labels, 0),
                         self.pl['is_training_pl']: is_training, }
            _, loss_val, pred_val, _ = self.sess.run([self.op['train_op'],
                                                   self.op['loss'],
                                                   self.op['pred'],
                                                   self.increment_global_step],
                                                   feed_dict=feed_dict)
            pred_val = np.argmax(pred_val, 2)
            correct = np.sum(pred_val == labels)
            self.cumsum_train_correct += correct
            self.cumsum_train += points_and_features.shape[0]
            logging.info(" -- Cum. Accuracy: %.2f %%" % (100*self.cumsum_train_correct/self.cumsum_train))
            logging.info(" -- Curr. Accuracy: %.2f %%" % (100*correct/points_and_features.shape[0]))
            logging.info(" -- Curr. Loss: %.5f" % loss_val)

    def test(self, test_ds, save=True):
        is_training = False
        feed_dict = {self.pl['pointclouds_pl']: np.expand_dims(test_ds.points_and_features, 0),
                     self.pl['labels_pl']: np.expand_dims(test_ds.labels, 0),
                     self.pl['is_training_pl']: is_training, }
        pred_val = self.sess.run([self.op['pred']], feed_dict=feed_dict)

        pred_val_max = np.argmax(pred_val[0], 2)
        for cl in np.unique(pred_val_max):
            elem_cnt = np.count_nonzero(pred_val_max == cl)
            logging.debug("Class %d: %d points (%.2f %%)" % (cl, elem_cnt, 100*elem_cnt/len(test_ds)))
        correct = np.sum(pred_val_max == test_ds.labels)
        self.cumsum_test_correct += correct
        self.cumsum_test += len(test_ds)
        logging.info(" -- Cum. Accuracy: %.2f %%" % (100*self.cumsum_test_correct / self.cumsum_test))
        logging.info(" -- Curr. Accuracy: %.2f %%" % (100*correct / len(test_ds)))
        if save:
            test_ds.save_with_new_classes(r'/data/lwiniwar/02_temp/alsNet_results/%s' % test_ds.filename,
                                          pred_val_max)

    def test_chunk(self, points_and_features, labels, save, names):
        is_training = False
        feed_dict = {self.pl['pointclouds_pl']: np.expand_dims(points_and_features, 0),
                     self.pl['labels_pl']: np.expand_dims(labels, 0),
                     self.pl['is_training_pl']: is_training, }
        pred_val, probs = self.sess.run([self.op['pred'], self.op['softmax']], feed_dict=feed_dict)

        pred_val_max = np.argmax(pred_val[0], 1)
        for cl in np.unique(pred_val_max):
            elem_cnt = np.count_nonzero(pred_val_max == cl)
            logging.debug("Class %d: %d points (%.2f %%)" % (cl, elem_cnt, 100 * elem_cnt /
                                                             points_and_features.shape[0]))
        if any(labels):
            correct = np.sum(pred_val_max == labels)
            self.cumsum_test_correct += correct
            self.cumsum_test += points_and_features.shape[0]
            logging.info(" -- Cum. Accuracy: %.2f %%" % (100 * self.cumsum_test_correct / self.cumsum_test))
            logging.info(" -- Curr. Accuracy: %.2f %%" % (100 * correct / points_and_features.shape[0]))
            if self.logger:
                self.logger.valid_points_seen.append((self.cumsum_train + points_and_features.shape[0]) * 1e-6)
                self.logger.valid_points_acc.append(100 * correct / points_and_features.shape[0])
                self.logger.valid_points_cumacc.append(100 * self.cumsum_test_correct / self.cumsum_test)
                cm = np.zeros((self.num_classes, self.num_classes))
                for a, p in zip(labels, pred_val_max):
                    cm[a][p] += 1
                cm /= len(labels)
                del_i = [i for i in range(self.num_classes) if i not in [2,3,4,5,6]]
                cm = np.delete(cm, del_i, 0)
                cm = np.delete(cm, del_i, 1)
                self.logger.valid_confusion.append(cm)
        if save:
            if any(labels):
                Dataset.Save(save, points_and_features, names, labels, pred_val_max, probs[0])
            else:
                Dataset.Save(save, points_and_features, names, new_classes=pred_val_max, probs=probs[0])


    def save_model(self, path_to_save):
        with self.graph.as_default():
            saver = tf.train.Saver(keep_checkpoint_every_n_hours=1)
            saver.save(self.sess, path_to_save)

    def load_from_file(self, meta_path_to_load):
        sess = self.sess


        with self.graph.as_default():
            saver = tf.train.import_meta_graph(meta_path_to_load)
            saver.restore(sess, tf.train.latest_checkpoint(os.path.dirname(meta_path_to_load)))
            #print(sess.run(tf.report_uninitialized_variables()))
            graph = tf.get_default_graph()
            self.pl['pointclouds_pl'] = graph.get_tensor_by_name('pointcloud_in:0')
            self.pl['labels_pl'] = graph.get_tensor_by_name('labels:0')

            self.pl['is_training_pl'] = graph.get_tensor_by_name('is_training:0')

            self.op['train_op'] = graph.get_operation_by_name('train')
            self.op['loss'] = graph.get_tensor_by_name('loss/value:0')
            self.op['pred'] = graph.get_tensor_by_name('fc2/net:0')
            self.op['softmax'] = graph.get_tensor_by_name('softmax:0')


        #self.sess = tf.Session(graph=graph)
        #self.sess.run(tf.report_uninitialized_variables())


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--inFiles', default='./*.laz', help='input files (wildcard supported) [default: ./*.laz]')
    parser.add_argument('--trainNum', default=2, type=int, help='How many files to use for training [default: 2]')
    parser.add_argument('--testNum', default=2, type=int, help='How many files to use for testing [default: 2]')
    args = parser.parse_args()

    alsNetInstance = AlsNetContainer('log')
    alsNetInstance.prepare(num_feat=1, num_classes=17)
    files = glob.glob(args.inFiles)
    random.shuffle(files)
    #files = files[::-1]
    train_num = args.trainNum
    test_num = args.testNum
    logging.info("--TRAINING--")
    idx = 0
    trainidx = 0
    while idx < train_num:
        file = files[trainidx]
        train_ds = Dataset(file)
        if len(train_ds) > 6000000:
            logging.info("File skipped (%s, %d points)" % (train_ds.filename, len(train_ds)))
            trainidx += 1
            continue
        logging.info("File %s/%s (%s, %d points)" % (idx+1, train_num, train_ds.filename, len(train_ds)))
        alsNetInstance.train(train_ds)
        trainidx += 1
        idx += 1

    logging.info("--TESTING--")
    testidx = -2  # to repeat the last 2
    while idx-train_num < test_num:
        file = files[testidx + trainidx]
        test_ds = Dataset(file)
        if len(test_ds) > 6000000:
            logging.info("File skipped (%s, %d points)" % (test_ds.filename, len(test_ds)))
            testidx += 1
            continue
        logging.info("File %s/%s (%s, %d points)" % (idx-train_num+1, test_num, test_ds.filename, len(test_ds)))
        alsNetInstance.test(test_ds)
        testidx += 1
        idx += 1
