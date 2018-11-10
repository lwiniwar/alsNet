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
from sklearn.metrics import confusion_matrix

def exclude_validation_loss(labels, logits):
    weights = tf.where((labels >= 10), np.zeros(labels.shape), np.ones(labels.shape))
    classify_loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits, scope='loss', weights=weights)
    return classify_loss

#Disable TF debug messages
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

default_arch =[
    {
        'npoint': 8192,
        'radius': 0.25,
        'nsample': 16,
        'mlp': [64, 128, 128],
        'pooling': 'max',
        'mlp2': None,
        'reverse_mlp': [128,64]
    },
    {
        'npoint': 4096,
        'radius': 0.5,
        'nsample': 16,
        'mlp': [128, 256, 256],
        'pooling': 'max',
        'mlp2': None,
        'reverse_mlp': [256,256]
    },
    {
        'npoint': 2048,
        'radius': 1,
        'nsample': 16,
        'mlp': [128, 256, 256],
        'pooling': 'max',
        'mlp2': None,
        'reverse_mlp': [256,256]
    },
    {
        'npoint': 512,
        'radius': 2,
        'nsample': 32,
        'mlp': [128, 512, 256],
        'pooling': 'max',
        'mlp2': None,
        'reverse_mlp': [256,256]
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
    arch = importlib.import_module(args.archFile).arch if args.archFile else default_arch

    print("Loading files")
    train_data = np.loadtxt(inFile)
    train_coords = train_data[:, 0:3]
    train_labels = train_data[:, 3]

    val_data = np.loadtxt(valFile)
    val_coords = val_data[:, 0:3]
    val_labels = val_data[:, 3] + 20

    test_data = np.loadtxt(testFile)
    test_coords = test_data[:, 0:3]
    test_labels = test_data[:, 3] + 10

    all_coords = np.vstack((train_coords, val_coords, test_coords))
    all_labels = np.expand_dims(np.hstack((train_labels, val_labels, test_labels)), -1)

    minx = np.min(all_coords[:, 0])
    miny = np.min(all_coords[:, 1])
    minz = np.min(all_coords[:, 2])
    maxx = np.max(all_coords[:, 0])
    maxy = np.max(all_coords[:, 1])
    maxz = np.max(all_coords[:, 2])

    #centersX = np.arange(minx, maxx, args.stepX)
    #centersY = np.arange(miny, maxy, args.stepY)
    #centersZ = np.arange(minz, maxz, args.stepZ)
    #num_batches = len(centersX) * len(centersY) * len(centersZ)
    #print("Will work on {} batches".format(num_batches))

    #print("Building kD tree for batch extraction")
    #tree = KDTree(all_coords, leafsize=100)

    print("Extracting batches as random subsets")
    idx_perm = np.random.permutation(np.arange(all_coords.shape[0]))



    inst = AlsNetContainer(num_feat=0, num_classes=3*10, num_points=200000, output_base=args.outDir, score_sample=10,
                           arch=arch,
                           learning_rate=lr,
                           dropout=0.55,
                           loss_fn=exclude_validation_loss)

    if args.continueModel is not None:
        inst.load_model(args.continueModel)
    else:
        inst.create_graph()

    logg = Logger(outfile=os.path.join(args.outDir, 'alsNet-log.html'),
                  inst=inst,
                  training_files=[File(inFile),File(testFile), File(valFile)])

    for j in range(args.multiTrain):
        if j > 0:
            # test current performance
            new_classes = np.zeros(all_labels.shape) + 50
            curr_eval_batch = 0
            for centerX in centersX:
                for centerY in centersY:
                    for centerZ in centersZ:
                        curr_eval_batch += 1
                        print("Evaluating batch {}/{}".format(curr_eval_batch, num_batches))
                        center = [centerX, centerY, centerZ]
                        _, idx = tree.query(center, k=200000)
                        probs = inst.predict_probability(all_coords[idx, :])
                        new_classes[idx] = np.argmax(probs, axis=2).T
            cm = confusion_matrix(all_labels[all_labels >= 20] - 20, new_classes[all_labels>=20], range(10))
            inst.eval_history.add_history_step(cm, inst._train_points_seen, 0)
            logg.save()

        curr_training_batch = 0
        for centerX in centersX:
            for centerY in centersY:
                for centerZ in centersZ:
                    curr_training_batch += 1
                    print("Training batch {}/{}".format(curr_training_batch, num_batches))
                    center = [centerX, centerY, centerZ]
                    _, idx = tree.query(center, k=200000)
                    curr_epoch_coords = all_coords[idx, :]
                    curr_epoch_labels = all_labels[idx, :]
                    inst.fit_one_epoch(curr_epoch_coords, curr_epoch_labels[:, 0])
                    np.savetxt(os.path.join(args.outDir,'sample_output_%.0f_%.0f_%.0f.xyz') % (centerX, centerY, centerZ), curr_epoch_coords)
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

    parser.add_argument('--archFile', default="", type=str,
                       help='architecture file to import [default: default architecture]')
    # parser.add_argument('--testList', help='list with files to test on')
    # parser.add_argument('--gpuID', default=None, help='which GPU to run on (default: CPU only)')
    args = parser.parse_args()
    main(args)
