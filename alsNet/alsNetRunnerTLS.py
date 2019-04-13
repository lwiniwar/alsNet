import os, sys
import numpy as np
import argparse
from alsNetRefactored import AlsNetContainer
from alsNetLogger2 import Logger
from dataset import Dataset
import tensorflow as tf
import importlib
from collections import namedtuple

from scipy.spatial import KDTree
from sklearn.metrics import confusion_matrix, f1_score

def exclude_validation_loss(labels, logits):
    weights = tf.where((labels >= 10), np.zeros(labels.shape), np.ones(labels.shape))
    classify_loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits, scope='loss', weights=weights)
    return classify_loss

def exclude_validation_loss_weigh_with_priors(labels, logits):
    priors = np.array([0, 931952, 2520358, 1692719, 35802, 894213, 549558, 10019, 47334] ) # from training data
    priors = priors/np.sum(priors)
    priors = 1/np.sqrt(priors)  # use the sqrt inverse of the class histogram as weights as suggested by stefan schmohl
    priors[0] = 0  # weight of class 0 = 0
    mask = tf.greater(labels, 10)
    zeros = tf.zeros_like(labels)
    labels_remap = tf.where(mask, zeros, labels)
    weights = tf.gather(priors, labels_remap)
    classify_loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits, scope='loss', weights=weights)
    return classify_loss

#Disable TF debug messages
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

default_arch =[
   # {
   #     'npoint': 8192*2,
   #     'radius': 0.01,
   #     'nsample': 16,
   #     'mlp': [64, 128, 128],
   #     'pooling': 'max_and_avg',
   #     'mlp2': None,
   #     'reverse_mlp': [128,128]
   # },
   # {
   #     'npoint': 4096*2,
   #     'radius': 0.05,
   #     'nsample': 16,
   #     'mlp': [128, 256, 256],
   #     'pooling': 'max_and_avg',
   #     'mlp2': None,
   #     'reverse_mlp': [256,256]
   # },
   # {
   #     'npoint': 2048,
   #     'radius': 0.1,
   #     'nsample': 16,
   #     'mlp': [256, 512, 512],
   #     'pooling': 'max_and_avg',
   #     'mlp2': None,
   #     'reverse_mlp': [512,512]
   # },
    {
        'npoint': 2048,
        'radius': 0.5,
        'nsample': 32,
        'mlp': [256, 512, 512],
        'pooling': 'max_and_avg',
        'mlp2': None,
        'reverse_mlp': [512,512]
    },
    {
        'npoint': 512,
        'radius': 1,
        'nsample': 64,
        'mlp': [512, 1024, 1024],
        'pooling': 'max_and_avg',
        'mlp2': None,
        'reverse_mlp': [1024,1024]
    },
    {
        'npoint': 256,
        'radius': 2,
        'nsample': 64,
        'mlp': [512, 1024, 1024],
        'pooling': 'max_and_avg',
        'mlp2': None,
        'reverse_mlp': [1024,1024]
    }, ]

File = namedtuple('File', 'file')

def main(args):
    #import os
    #os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    inFile = args.inFile
    testFile = args.testFile
    valFile = args.valFile
    lr = args.learningRate
    normalize_vals = args.normalize == 1
    eval_only = args.evaluate_only == 1

    arch = importlib.import_module(args.archFile).arch if args.archFile else default_arch

    print("Loading files")
    train_data = np.loadtxt(inFile)
    train_coords = train_data[:, 0:3]
    train_labels = train_data[:, 3]

    val_data = np.loadtxt(valFile)
    val_coords = val_data[:, 0:3]
    val_labels = val_data[:, 3] + 10

    test_data = np.loadtxt(testFile)
    test_coords = test_data[:, 0:3]
    test_labels = test_data[:, 3] + 20

    all_coords = np.vstack((train_coords, val_coords, test_coords))
    all_labels = np.expand_dims(np.hstack((train_labels, val_labels, test_labels)), -1)


    num_points = all_coords.shape[0]
    num_batches = num_points // 200000 + 1
    print("Will work on {} batches".format(num_batches))




    inst = AlsNetContainer(num_feat=0, num_classes=3*10, num_points=200000, output_base=args.outDir, score_sample=10,
                           arch=arch,
                           learning_rate=lr,
                           dropout=0.55,
                           loss_fn=exclude_validation_loss_weigh_with_priors,
                           gpu_id=0)

    if args.continueModel is not None:
        inst.load_model(args.continueModel)
    else:
        inst.create_graph()

    logg = Logger(outfile=os.path.join(args.outDir, 'alsNet-log.html'),
                  inst=inst,
                  training_files=[File(inFile),File(testFile), File(valFile)])

    for j in range(args.multiTrain):

        print("Extracting batches as random subsets")
        idx_perm = np.random.permutation(num_points)
        all_coords_perm = all_coords[idx_perm, :]
        all_labels_perm = all_labels[idx_perm, :]


        #remove classes 4,7,8 as requested by ash
        all_coords_perm = all_coords_perm[all_labels_perm[:, 0] != 4, :]
        all_labels_perm = all_labels_perm[all_labels_perm[:, 0] != 4, :]
        all_coords_perm = all_coords_perm[all_labels_perm[:, 0] != 7, :]
        all_labels_perm = all_labels_perm[all_labels_perm[:, 0] != 7, :]
        all_coords_perm = all_coords_perm[all_labels_perm[:, 0] != 8, :]
        all_labels_perm = all_labels_perm[all_labels_perm[:, 0] != 8, :]

        all_labels_perm[all_labels_perm[:, 0] == 15, :] = 16
        all_labels_perm[all_labels_perm[:, 0] == 14, :] = 15
        all_labels_perm[all_labels_perm[:, 0] == 25, :] = 26
        all_labels_perm[all_labels_perm[:, 0] == 24, :] = 25

        num_points = all_coords_perm.shape[0]
        num_batches = num_points // 200000 + 1

        if eval_only:
            print("Evaluating all batches, extracting validation samples and creating confusion matrix")
            result_labels = np.zeros(all_labels_perm.shape)

            for curr_eval_batch in range(num_batches):
                idx_start = curr_eval_batch * 200000
                idx_end = (curr_eval_batch + 1) * 200000
                if idx_end >= num_points:
                    idx_start -= (idx_end - num_points + 1)
                    idx_end = num_points - 1


                curr_epoch_coords = all_coords_perm[idx_start:idx_end]

                print("Evaluating batch {}/{}".format(curr_eval_batch, num_batches))
                probs = inst.predict_probability(curr_epoch_coords)
                new_classes = np.argmax(probs, axis=2).T
                result_labels[idx_start:idx_end] = new_classes

            cm = confusion_matrix(all_labels_perm[all_labels_perm >= 20] - 20,
                                  result_labels[all_labels_perm >= 20], range(10))
            loc = (all_labels_perm >= 20)
            val_coords = all_coords_perm[loc[:, 0], :]
            val_labels = np.expand_dims(all_labels_perm[loc] - 20, -1)
            val_predictions = np.expand_dims(result_labels[loc], -1)

            print(val_coords.shape)
            print(val_labels.shape)
            print(val_predictions.shape)
            np.savetxt(os.path.join(args.outDir, 'eval_points.xyz'), np.hstack((val_coords, val_labels, val_predictions)))
            np.savetxt(os.path.join(args.outDir, 'eval_cm.txt'), cm)
            prec = np.zeros(10)
            sens = np.zeros(10)
            for i in range(10):
                prec[i] = cm[i,i]/np.sum(cm[:, i])
                sens[i] = cm[i,i]/np.sum(cm[i, :])
            f1 = 2 * prec * sens / (prec + sens)
            acc = np.diag(cm) / np.sum(cm, axis=0)
            oa = np.nansum(acc * np.sum(cm, axis=0) / np.sum(cm))
            np.savetxt(os.path.join(args.outDir, 'eval_prec.txt'), prec)
            np.savetxt(os.path.join(args.outDir, 'eval_sens.txt'), sens)
            np.savetxt(os.path.join(args.outDir, 'eval_f1.txt'), f1)
            np.savetxt(os.path.join(args.outDir, 'eval_acc.txt'), acc)
            print("Overall accuracy: {:.3f}%".format(oa*100))
            print("F1-Scores: \n", f1)


            exit(0)



        for curr_training_batch in range(num_batches):
            idx_start = curr_training_batch * 200000
            idx_end = (curr_training_batch + 1) * 200000
            if idx_end >= num_points:
                idx_start -= (idx_end-num_points+1)
                idx_end = num_points-1

            print("Training batch {}/{}".format(curr_training_batch, num_batches))

            curr_epoch_coords = all_coords_perm[idx_start:idx_end]
            curr_epoch_labels = all_labels_perm[idx_start:idx_end]

            inst.fit_one_epoch(curr_epoch_coords, curr_epoch_labels[:, 0])
            np.savetxt(os.path.join(args.outDir,'sample_output_%d.xyz') % (curr_training_batch), curr_epoch_coords)
            logg.save()

            if curr_training_batch % 4 == 1:  # we need to have had _some_ training, so we take mod 1
                # pick a random batch for validation
                curr_eval_batch = np.random.choice(num_batches, 1)[0]
                idx_start = curr_eval_batch * 200000
                idx_end = (curr_eval_batch + 1) * 200000
                if idx_end >= num_points:
                    idx_start -= (idx_end - num_points + 1)
                    idx_end = num_points - 1
                print("Evaluating on batch {}".format(curr_eval_batch, num_batches))

                curr_epoch_coords = all_coords_perm[idx_start:idx_end]
                curr_epoch_labels = all_labels_perm[idx_start:idx_end]

                probs = inst.predict_probability(curr_epoch_coords)
                new_classes = np.argmax(probs, axis=2).T
                cm = confusion_matrix(curr_epoch_labels[np.logical_and(curr_epoch_labels >= 10, curr_epoch_labels < 20)] - 10,
                                      new_classes[np.logical_and(curr_epoch_labels >= 10, curr_epoch_labels < 20)], range(10))
                inst.eval_history.add_history_step(cm, inst._train_points_seen, 0)
                logg.save()

        inst.save_model(os.path.join(args.outDir, 'models', 'model_%d' % (j), 'alsNet.ckpt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inFile', help='training text file, must be X/Y/Z/Class')
    parser.add_argument('--testFile', help='test text file, must be X/Y/Z/Class')
    parser.add_argument('--valFile', help='validation text file, must be X/Y/Z/Class')
    parser.add_argument('--outDir', required=True, help='directory to write html log to')
    # parser.add_argument('--multiclass', default=True, type=bool, help='label into multiple classes ' +
    #                                                                  '(not only ground/nonground) [default: True]')
    parser.add_argument('--multiTrain', default=1, type=int,
                       help='how often to feed the whole training dataset [default: 1]')
    parser.add_argument('--stepX', default=1, type=float,
                       help='batch step size in X [default: 1]')
    parser.add_argument('--stepY', default=1, type=float,
                       help='batch step size in Y [default: 1]')
    parser.add_argument('--stepZ', default=1, type=float,
                       help='batch step size in Z [default: 1]')
    parser.add_argument('--learningRate', default=0.001, type=float,
                       help='learning rate [default: 0.001]')
    parser.add_argument('--continueModel', default=None, type=str,
                        help='continue training an existing model [default: start new model]')
    parser.add_argument('--normalize', default=1, type=int,
                        help='normalize fields and coordinates [default: 1][1/0]')
    parser.add_argument('--evaluate_only', default=0, type=int,
                        help='evaluate the data [default: 1][1/0]')

    parser.add_argument('--archFile', default="", type=str,
                       help='architecture file to import [default: default architecture]')
    # parser.add_argument('--testList', help='list with files to test on')
    # parser.add_argument('--gpuID', default=None, help='which GPU to run on (default: CPU only)')
    args = parser.parse_args()
    main(args)
