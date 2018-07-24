import os, sys
import numpy as np
import argparse
from alsNetRefactored import AlsNetContainer, simple_loss, fp_high_loss
from dataset import Dataset
from sklearn.model_selection import RandomizedSearchCV

#Disable TF debug messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

arch1 =[
    {
        'npoint': 512,
        'radius': 1,
        'nsample': 128,
        'mlp': [512, 512, 1024],
        'pooling': 'max',
        'mlp2': None,
        'reverse_mlp': [128,128]
    },
    {
        'npoint': 256,
        'radius': 5,
        'nsample': 64,
        'mlp': [128, 128, 256],
        'pooling': 'max',
        'mlp2': None,
        'reverse_mlp': [256,256]
    },
    {
        'npoint': 128,
        'radius': 15,
        'nsample': 64,
        'mlp': [128, 128, 256],
        'pooling': 'max',
        'mlp2': None,
        'reverse_mlp': [256,256]
    }, ]
arch2 =[
    {
        'npoint': 1024,
        'radius': 1,
        'nsample': 128,
        'mlp': [512, 512, 1024],
        'pooling': 'max',
        'mlp2': None,
        'reverse_mlp': [1024,512]
    },
    {
        'npoint': 512,
        'radius': 5,
        'nsample': 64,
        'mlp': [128, 128, 256],
        'pooling': 'max',
        'mlp2': None,
        'reverse_mlp': [256,256]
    },
    {
        'npoint': 256,
        'radius': 15,
        'nsample': 64,
        'mlp': [128, 128, 256],
        'pooling': 'max',
        'mlp2': None,
        'reverse_mlp': [256,256]
    }, ]
arch3 =[
    {
        'npoint': 256,
        'radius': 1,
        'nsample': 128,
        'mlp': [256, 256, 256],
        'pooling': 'max',
        'mlp2': None,
        'reverse_mlp': [256,256]
    },
    {
        'npoint': 128,
        'radius': 5,
        'nsample': 64,
        'mlp': [128, 128, 256],
        'pooling': 'max',
        'mlp2': None,
        'reverse_mlp': [256,256]
    },
    {
        'npoint': 64,
        'radius': 15,
        'nsample': 64,
        'mlp': [128, 128, 256],
        'pooling': 'max',
        'mlp2': None,
        'reverse_mlp': [256,256]
    }, ]


param_distr = {
    'arch': [arch1, arch2, arch3],
    'learning_rate': [0.1, 0.01, 0.001],
    'dropout': [0.55, 0.6, 0.65],
    'loss_fn': [simple_loss, fp_high_loss]
}

def main(args):
    inlist = args.inList
    threshold = args.threshold
    #train_size = args.trainSize

    with open(inlist, "rb") as f:
        _ = f.readline()  # remove header
        rest = f.readlines()

    datasets = []
    all_ds = []
    for line in rest:
        line = line.decode('utf-8')
        linespl = line.split(",")
        dataset_path = os.path.join(os.path.dirname(inlist), linespl[0])
        if float(linespl[1]) < threshold:
            datasets.append(dataset_path)
        all_ds.append(dataset_path)

    np.random.shuffle(datasets)
    datasets_th = []
    for idx, dataset in enumerate(datasets):
        print("Loading dataset %s of %s" % (idx+1, len(datasets)))
        ds = Dataset(dataset, load=False)
        datasets_th.append(ds)
    print("%s datasets loaded." % len(datasets_th))
    sys.stdout.flush()

    rnd_search = RandomizedSearchCV(AlsNetContainer(num_feat=3, num_classes=30, num_points=200000,
                                                    output_base=args.outDir, score_sample=10),
                                    param_distr,
                                    n_iter=50,
                                    random_state=42,
                                    verbose=2,
                                    n_jobs=1)
    rnd_search.fit(datasets_th)
    print(rnd_search.best_params_)

    #inst = AlsNetContainer(num_points=200000, num_classes=30, num_feat=3, arch=arch,
    #                       output_dir=args.outDir, dropout=args.dropout)
    #logg = Logger(outfile=os.path.join(args.outDir, 'alsNet-log.html'),
    #              inst=inst,
    #              training_files=datasets)
    #for i in range(len(datasets)//train_size):
    #    if i > 0:
    #        test_ds = datasets[i*train_size+1]
    #        inst.test_single(test_ds,
    #                         save_to=os.path.join(args.outDir, os.path.basename(test_ds).replace(".la", "_test.la")),
    #                         save_prob=True)
    #    print("Training datasets %s to %s (%s total)" % (i*train_size,
    #                                                     min((i+1)*train_size, len(datasets)),
    #                                                     len(datasets)))
    #    inst.fit(datasets[i*train_size:min((i+1)*train_size, len(datasets))], new_session=False)
    #    logg.save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inList', help='input text file, must be csv with filename;stddev;...')
    parser.add_argument('--threshold', type=float, help='upper threshold for class stddev')
    parser.add_argument('--outDir', required=True, help='directory to write html log to')
    # parser.add_argument('--multiclass', default=True, type=bool, help='label into multiple classes ' +
    #                                                                  '(not only ground/nonground) [default: True]')
    # parser.add_argument('--multiTrain', default=1, type=int,
    #                    help='how often to feed the whole training dataset [default: 1]')
    # parser.add_argument('--testList', help='list with files to test on')
    # parser.add_argument('--gpuID', default=None, help='which GPU to run on (default: CPU only)')
    args = parser.parse_args()
    main(args)