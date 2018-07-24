import glob
from argparse import ArgumentParser
from dataset import Dataset
from scipy.spatial import kdtree
import os
import logging

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s [%(levelname)s]: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')


def main(in_files, out_file, write_probs=True):
    input_files = []
    for filepattern in in_files:
        for file in glob.glob(filepattern):
            input_files.append(file)

    logging.info("Found %d files to merge" % len(input_files))

    overall_tree = kdtree.KDTree()
    for file in input_files:
        ds = Dataset(file)



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--inFiles',
                        default=[],
                        required=True,
                        help='input files (wildcard supported)',
                        action='append')
    parser.add_argument('--outFile', required=True, help='path to write output to')
    parser.add_argument('--writeProbs', default=True, type=bool, help='write class probabilities')
    args = parser.parse_args()

    main(args.inFiles, args.outFile, args.writeProbs)