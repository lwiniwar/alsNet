import os, sys
import numpy as np
import argparse
from alsNetRefactored import AlsNetContainer, simple_loss, fp_high_loss
from alsNetLogger2 import Logger
from dataset import Dataset
import importlib

#Disable TF debug messages
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

arch =[
    {
        'npoint': 8192,
        'radius': 1,
        'nsample': 16,
        'mlp': [64, 128, 128],
        'pooling': 'max',
        'mlp2': None,
        'reverse_mlp': [128,64]
    },
    {
        'npoint': 4096,
        'radius': 2,
        'nsample': 16,
        'mlp': [128, 256, 256],
        'pooling': 'max',
        'mlp2': None,
        'reverse_mlp': [256,256]
    },
    {
        'npoint': 2048,
        'radius': 5,
        'nsample': 16,
        'mlp': [128, 256, 256],
        'pooling': 'max',
        'mlp2': None,
        'reverse_mlp': [256,256]
    },
    {
        'npoint': 512,
        'radius': 15,
        'nsample': 32,
        'mlp': [128, 512, 256],
        'pooling': 'max',
        'mlp2': None,
        'reverse_mlp': [256,256]
    }, ]


def main(args):
    #import os
    #os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    inlist = args.inList
    threshold = args.threshold
    train_size = args.trainSize
    arch = importlib.import_module(args.archFile).arch if args.archFile else arch
    lr = args.learningRate
    normalize_vals = args.normalize == 1

    with open(inlist, "rb") as f:
        _ = f.readline()  # remove header
        rest = f.readlines()

    datasets = []
    all_ds = []
    for line in rest:
        line = line.decode('utf-8')
        linespl = line.split(",")
        dataset_path = os.path.join(os.path.dirname(inlist), linespl[0])
        if float(linespl[1]) < threshold and float(linespl[6]) > args.minBuild:
            datasets.append(dataset_path)
        all_ds.append(dataset_path)

    np.random.shuffle(datasets)
    datasets_th = []
    for idx, dataset in enumerate(datasets):
        print("Loading dataset %s of %s (%s)" % (idx+1, len(datasets), os.path.basename(dataset)))
        ds = Dataset(dataset, load=False, normalize=normalize_vals)
        datasets_th.append(ds)
    print("%s datasets loaded." % len(datasets_th))
    sys.stdout.flush()

    inst = AlsNetContainer(num_feat=3, num_classes=30, num_points=200000, output_base=args.outDir, score_sample=10,
                           arch=arch,
                           learning_rate=lr,
                           dropout=0.55,
                           loss_fn=simple_loss if args.lossFn == "simple" else fp_high_loss)

    if args.continueModel is not None:
        inst.load_model(args.continueModel)

    logg = Logger(outfile=os.path.join(args.outDir, 'alsNet-log.html'),
                  inst=inst,
                  training_files=datasets_th)

    for j in range(args.multiTrain):
        for i in range(len(datasets_th)//train_size):
            if i > 0 and i*train_size+1 < len(datasets_th):
                test_ds = datasets_th[i*train_size+1]
                inst.test_single(test_ds,
                                 save_to=os.path.join(args.outDir, os.path.basename(test_ds.file).replace(".la", "_test.la")),
                                 save_prob=True)
            print("Training datasets %s to %s (%s total)" % (i*train_size,
                                                             min((i+1)*train_size, len(datasets_th)),
                                                             len(datasets_th)))
            inst.fit(datasets_th[i*train_size:min((i+1)*train_size, len(datasets_th))], new_session=False)
            logg.save()
        inst.save_model(os.path.join(args.outDir, 'models', 'model_%d_%d' % (j, i), 'alsNet.ckpt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inList', help='input text file, must be csv with filename;stddev;...')
    parser.add_argument('--threshold', type=float, help='upper threshold for class stddev')
    parser.add_argument('--minBuild', type=float, help='lower threshold for buildings class [0-1]')
    parser.add_argument('--outDir', required=True, help='directory to write html log to')
    # parser.add_argument('--multiclass', default=True, type=bool, help='label into multiple classes ' +
    #                                                                  '(not only ground/nonground) [default: True]')
    parser.add_argument('--multiTrain', default=1, type=int,
                       help='how often to feed the whole training dataset [default: 1]')
    parser.add_argument('--trainSize', default=1, type=int,
                       help='how many plots to train at once [default: 1]')
    parser.add_argument('--learningRate', default=0.001, type=float,
                       help='learning rate [default: 0.001]')
    parser.add_argument('--archFile', default="", type=str,
                       help='architecture file to import [default: default architecture]')
    parser.add_argument('--continueModel', default=None, type=str,
                        help='continue training an existing model [default: start new model]')
    parser.add_argument('--lossFn', default='simple', type=str,
                        help='loss function to use [default: simple][simple/fp_high]')
    parser.add_argument('--normalize', default=1, type=int,
                        help='normalize fields and coordinates [default: 1][1/0]')
    # parser.add_argument('--testList', help='list with files to test on')
    # parser.add_argument('--gpuID', default=None, help='which GPU to run on (default: CPU only)')
    args = parser.parse_args()
    main(args)
