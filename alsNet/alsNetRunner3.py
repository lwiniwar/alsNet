import os
import numpy as np
import argparse
from alsNetRefactored import AlsNetContainer
from alsNetLogger2 import Logger

#Disable TF debug messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

arch =[
    {
        'npoint': 4096,
        'radius': 1,
        'nsample': 32,
        'mlp': [128, 128, 128],
        'pooling': 'max',
        'mlp2': None,
        'reverse_mlp': [128,128]
    },
    {
        'npoint': 2048,
        'radius': 5,
        'nsample': 64,
        'mlp': [128, 128, 256],
        'pooling': 'max',
        'mlp2': None,
        'reverse_mlp': [256,256]
    },
    {
        'npoint': 1024,
        'radius': 15,
        'nsample': 64,
        'mlp': [128, 128, 256],
        'pooling': 'max',
        'mlp2': None,
        'reverse_mlp': [256,256]
    }, ]

def main(args):
    inlist = args.inList
    threshold = args.threshold
    train_size = args.trainSize

    with open(inlist, "rb") as f:
        _ = f.readline()  # remove header
        rest = f.readlines()

    datasets = []
    for line in rest:
        line = line.decode('utf-8')
        linespl = line.split(",")
        if float(linespl[1]) < threshold:
            datasets.append(os.path.join(os.path.dirname(inlist), linespl[0]))

    np.random.shuffle(datasets)
    inst = AlsNetContainer(num_points=200000, num_classes=30, num_feat=3, arch=arch,
                           output_dir=args.outDir, dropout=args.dropout)
    logg = Logger(outfile=os.path.join(args.outDir, 'alsNet-log.html'),
                  inst=inst,
                  training_files=datasets)
    for i in range(len(datasets)//train_size):
        if i > 0:
            test_ds = datasets[i*train_size+1]
            inst.test_single(test_ds,
                             save_to=os.path.join(args.outDir, os.path.basename(test_ds).replace(".la", "_test.la")),
                             save_prob=True)
        print("Training datasets %s to %s (%s total)" % (i*train_size,
                                                         min((i+1)*train_size, len(datasets)),
                                                         len(datasets)))
        inst.fit_file(datasets[i * train_size:min((i + 1) * train_size, len(datasets))], new_session=False)
        logg.save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inList', help='input text file, must be csv with filename;stddev;...')
    parser.add_argument('--threshold', type=float, help='upper threshold for class stddev')
    parser.add_argument('--trainSize', default=10, type=int, help='"batch" size for training [default: 10]')
    parser.add_argument('--dropout', default=0.5, type=float, help='probability to randomly drop a neuron ' +
                                                                   'in the last layer [default: 0.5]')
    parser.add_argument('--outDir', required=True, help='directory to write html log to')
    # parser.add_argument('--multiclass', default=True, type=bool, help='label into multiple classes ' +
    #                                                                  '(not only ground/nonground) [default: True]')
    # parser.add_argument('--multiTrain', default=1, type=int,
    #                    help='how often to feed the whole training dataset [default: 1]')
    # parser.add_argument('--testList', help='list with files to test on')
    # parser.add_argument('--gpuID', default=None, help='which GPU to run on (default: CPU only)')
    args = parser.parse_args()
    main(args)