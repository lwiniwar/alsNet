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

MAX_CLASSES = 30

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

    logging.info("Loading reference dataset")
    ref_ds = Dataset(ref_file)
    ref_points = ref_ds._xyz
    out_labels = ref_ds.labels

    prob_sums = np.zeros((ref_points.shape[0], MAX_CLASSES))
    prob_counts = np.zeros((ref_points.shape[0],))

    logging.info("Building 2D kD-Tree on the reference dataset")
    tree = ckdtree.cKDTree(ref_points[:, 0:2])  # only on 2D :D

    for fileidx, file in enumerate(input_files):
        logging.info("Processing file %d" % fileidx)
        ds = Dataset(file)
        points = np.hstack((ds.points_and_features, np.expand_dims(ds.labels, -1)))
        names = ds.names
        prob_ids_here = []
        prob_ids_ref = []
        for idx, name in enumerate(names):
            if name.startswith('prob_class'):
                prob_ids_here.append(idx+3)
                prob_ids_ref.append(int(name.split('prob_class')[-1]))

        for ptidx in range(points.shape[0]):
            xy = points[ptidx, 0:2]
            ref_ids = tree.query_ball_point(xy, r=0.0001, eps=0.0001)
            if len(ref_ids) > 1:
                ref_id = ref_ids[np.argmin(np.abs(ref_points[ref_ids, -1] - points[ptidx, 3]), axis=0)]
            elif len(ref_ids) == 0:
                logging.warn("Point not found: %s" % xy)
                continue
            else:
                ref_id = ref_ids[0]
            prob_counts[ref_id] += 1
            probs_here = points[ptidx, prob_ids_here]
            prob_sums[ref_id, prob_ids_ref] += probs_here
        del ds
        del points

    # clear memory
    ref_ds = None

    out_points = ref_points
    print(prob_counts)
    print(prob_sums[ref_id, :])

    #prob_avgs = prob_sums / prob_counts[:, np.newaxis]
    #print(prob_avgs)
    #print(prob_avgs[ref_id, :])
    new_max_class = np.zeros((ref_points.shape[0]))
    for i in range(ref_points.shape[0]):
        curr_point = prob_sums[i, :] / prob_counts[i]
        curr_point_max = np.argmax(curr_point)
        new_max_class[i] = curr_point_max
    #new_max_class = np.argmax(prob_avgs, axis=1)
    print(new_max_class)
    print(new_max_class[ref_id])
    print(out_labels[ref_id])
    #print(np.argmax(prob_avgs, axis=0))
    print(new_max_class.shape, out_labels.shape)
    #avg_feats = np.mean(prob_avgs, axis=1)
    #var_feats = np.std(prob_avgs, axis=1)

    #for line in range(ref_points.shape[0]):
    #    if line%10 == 0:
    #        #logging.info("Currently in line %d from %d" % (line, ref_points.shape[0]))
    #        pass
    #    curr_xyz = ref_points[line,:]
    #    multiples = tree.query_ball_point(curr_xyz, r=0.0001, eps=0.0001)
    #    #print(curr_xyz)
    #    #print(overall_points[multiples, 0:3])
    #    idx_processed += multiples
    #    if len(multiples) == 0:
    #        logging.info("Point missing: %s" % curr_xyz)
    #        continue
    #    out_points.append(curr_xyz)
    #    out_labels.append(overall_points[multiples[0], -1])
    #    out_meanvar.append(np.mean(var_feats))
    #    out_attrs.append(avg_feats)
    #    new_max_class.append(np.argmax(avg_feats[prob_cols], axis=0))
    #    #print(np.argmax(avg_feats[prob_cols], axis=0))
    #    #print(multiples, len(multiples))
    #    out_counts.append(len(multiples))

    #out_attrs = np.array(out_attrs)
    #out_counts = np.expand_dims(np.array(out_counts), -1)
    #out_meanvar = np.expand_dims(np.array(out_meanvar), -1)
    #names.append('meanvar')
    #names.append('count')
    ##attr_avg = np.sum(out_attrs, axis=1)/out_counts
    #out_labels = np.array(out_labels).astype(np.int)
    #Dataset.Save(out_file, np.hstack((out_points, out_attrs, out_meanvar, out_counts)), names, labels=out_labels, new_classes=new_max_class,
    #             )#addDims=addDims + ['meanvar', 'count'])
    #staticstics
    #pre_acc = np.count_nonzero(overall_points[:, estim_col] == overall_points[:, -1]) / overall_points.shape[0]
    pre_acc = 0
    post_acc = np.count_nonzero(new_max_class == out_labels) / len(out_points)
    post_cm = metrics.confusion_matrix(out_labels, new_max_class, labels=range(17))
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