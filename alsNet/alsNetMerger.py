import glob
from argparse import ArgumentParser
from dataset import Dataset
from scipy.spatial import ckdtree
from sklearn import metrics
import numpy as np
import os
import logging

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s [%(levelname)s]: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')


def main(in_files, ref_file, out_file, write_probs=True):
    input_files = []
    for filepattern in in_files:
        for file in glob.glob(filepattern):
            input_files.append(file)

    logging.info("Found %d files to merge" % len(input_files))

    overall_points = None

    out_points = []
    out_attrs = []
    out_counts = []
    out_meanvar = []
    out_labels = []
    new_max_class = []
    names = None

    for file in input_files:
        ds = Dataset(file)
        points = np.hstack((ds.points_and_features, np.expand_dims(ds.labels, -1)))
        if overall_points is None:
            overall_points = points
        else:
            overall_points = np.vstack((overall_points, points))
        names = ds.names

    ref_ds = Dataset(ref_file)
    ref_points = ref_ds._xyz

    prob_cols = []
    estim_col = None
    addDims=[]
    for idx, name in enumerate(names):
        if name.startswith('prob_class'):
            prob_cols.append(idx)
            addDims.append(name)
        if name == 'estim_class':
            estim_col = idx+3

    tree = ckdtree.cKDTree(overall_points[:, 0:3])
    idx_processed = []
    for line in range(ref_points.shape[0]):
        if line%10 == 0:
            #logging.info("Currently in line %d from %d" % (line, ref_points.shape[0]))
            pass
        curr_xyz = ref_points[line,:]
        multiples = tree.query_ball_point(curr_xyz, r=0.0001, eps=0.0001)
        #print(curr_xyz)
        #print(overall_points[multiples, 0:3])
        idx_processed += multiples
        if len(multiples) == 0:
            logging.info("Point missing: %s" % curr_xyz)
            continue
        out_points.append(curr_xyz)
        out_labels.append(overall_points[multiples[0], -1])
        avg_feats = np.mean(overall_points[multiples, 3:-1], axis=0)
        var_feats = np.std(overall_points[multiples, 3:-1], axis=0)
        out_meanvar.append(np.mean(var_feats))
        out_attrs.append(avg_feats)
        new_max_class.append(np.argmax(avg_feats[prob_cols], axis=0))
        #print(np.argmax(avg_feats[prob_cols], axis=0))
        #print(multiples, len(multiples))
        out_counts.append(len(multiples))

    out_attrs = np.array(out_attrs)
    out_counts = np.expand_dims(np.array(out_counts), -1)
    out_meanvar = np.expand_dims(np.array(out_meanvar), -1)
    names.append('meanvar')
    names.append('count')
    #attr_avg = np.sum(out_attrs, axis=1)/out_counts
    out_labels = np.array(out_labels).astype(np.int)
    Dataset.Save(out_file, np.hstack((out_points, out_attrs, out_meanvar, out_counts)), names, labels=out_labels, new_classes=new_max_class,
                 )#addDims=addDims + ['meanvar', 'count'])
    #staticstics
    pre_acc = np.count_nonzero(overall_points[:, estim_col] == overall_points[:, -1]) / overall_points.shape[0]
    post_acc = np.count_nonzero(new_max_class == out_labels) / len(out_points)
    post_cm = metrics.confusion_matrix(out_labels, new_max_class)
    post_prec = metrics.precision_score(out_labels, new_max_class, average=None)
    post_recall = metrics.recall_score(out_labels, new_max_class, average=None)
    post_f1 = metrics.f1_score(out_labels, new_max_class, average=None)
    np.savetxt(out_file + '_cm.txt', post_cm, fmt='%.4f')
    np.savetxt(out_file + '_prec.txt', post_prec, fmt='%.4f')
    np.savetxt(out_file + '_recall.txt', post_recall, fmt='%.4f')
    np.savetxt(out_file + '_f1.txt', post_f1, fmt='%.4f')
    logging.info("Finished. Pre-acc: %.3f | Post-acc: %.3f" % (pre_acc, post_acc))




if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--inFiles',
                        default=[],
                        required=True,
                        help='input files (wildcard supported)',
                        action='append')
    parser.add_argument('--refFile',
                        required=True,
                        help='File with all the output points present')
    parser.add_argument('--outFile', required=True, help='path to write output to')
    parser.add_argument('--writeProbs', default=True, type=bool, help='write class probabilities')
    args = parser.parse_args()

    main(args.inFiles, args.refFile, args.outFile, args.writeProbs)