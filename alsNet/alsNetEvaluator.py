import glob

from argparse import ArgumentParser
from alsNetRefactored import AlsNetContainer
from dataset import Dataset
import numpy as np
import os, sys
import logging
import importlib


# disable tensorflow debug information:
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s [%(levelname)s]: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')


def main(args):
    arch = importlib.import_module(args.arch).arch
    normalize = args.normalize
    model = AlsNetContainer(num_feat=3, num_classes=30, num_points=200000, output_base=args.outDir, arch=arch)
    logging.info("Loading pretrained model %s" % args.model)
    model.load_model(args.model)
    datasets = []
    if not os.path.exists(args.outDir):
        os.makedirs(args.outDir)
    for filepattern in args.inFiles:
        for file in glob.glob(filepattern):
            datasets.append(Dataset(file, load=False, normalize=normalize))
    total_acc = 0
    total_batch = 0
    for idx, dataset in enumerate(datasets):
        logging.info("Loading dataset %d / %d (%s)" % (idx, len(datasets), dataset.filename))
        acc = model.test_single(dataset,
                         save_to=os.path.join(args.outDir, os.path.basename(dataset.file).replace(".la", "_test.la")),
                         save_prob=True, unload=False)
        logging.info("Current test accuracy: %.2f%%" % (acc * 100.))
        meanxy = np.mean(dataset._xyz, axis=1)[0:2]
        with open(os.path.join(args.outDir, 'result.csv'), 'a') as out_stat_file:
            out_stat_file.write("%s, %.3f, %.3f, %.4f\n" % (dataset.file, meanxy[0], meanxy[1], acc) )
        dataset.unload()
        total_acc += acc
        total_batch += 1
        logging.info("Current avg test accuracy: %.2f%%" % ((total_acc/total_batch) * 100.))
        sys.stdout.flush()




if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--inFiles',
                        default=[],
                        required=True,
                        help='input files (wildcard supported)',
                        action='append')
    parser.add_argument('--model', required=True, help='tensorflow model ckpt file')
    parser.add_argument('--arch', required=True, help='python architecture file')
    parser.add_argument('--outDir', required=True, help='log and output directory')
    parser.add_argument('--normalize', default=1, type=int,
                        help='normalize fields and coordinates [default: 1][1/0]')
    args = parser.parse_args()

    main(args)