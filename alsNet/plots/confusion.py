from __future__ import print_function
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
import os
from sklearn.metrics import confusion_matrix
import glob
import sys
from alsNet.dataset import Dataset

class_names={
    0: 'Power',
    1: 'Low Veg.',
    2: 'Imp. Surf.',
    3: 'Car',
    4: 'Fence/Hedge',
    5: 'Roof',
    6: 'Facade',
    7: 'Shrub',
    8: 'Tree',
}
class_names={
    2: 'Ground',
    3: 'Low Veg.',
    4: 'Med. Veg.',
    5: 'High Veg.',
}
class_names={
    2: 'Ground',
    3: 'Low Veg.',
    4: 'Med. Veg.',
    5: 'High Veg.',
    6: 'Building',
    9: 'Water',
    -1: 'Other'
}

def get_cm_compressed(cm, keep_classes=(2, 3, 4, 5, 6, 9), delete=False):
    """
    Compresses a confusion matrix into the interesting columns/rows
    (careful, they are not ordered according to keep_classes, but the indices change!)
    and collects the rest in the last column/row
    :param cm: a 2D confusion matrix
    :param keep_classes: a set of classes to keep
    :param delete: delete rows from matrix after caluclation (default: False)
    :return:
    """
    coll_idx = cm.shape[0]
    cm_buf = np.append(cm, np.zeros((1, coll_idx)), axis=0)
    cm_buf = np.append(cm_buf, np.zeros((coll_idx + 1, 1)), axis=1)
    sum_idxs = [i for i in range(coll_idx) if i not in keep_classes]
    cm_buf[:, coll_idx] = np.sum(cm_buf[:, sum_idxs], axis=1)
    cm_buf[coll_idx, :] = np.sum(cm_buf[sum_idxs, :], axis=0)
    cm_buf[coll_idx, coll_idx] = np.sum(cm_buf[sum_idxs, -1])
    if delete:
        cm_buf = np.delete(cm_buf, sum_idxs, axis=0)
        cm_buf = np.delete(cm_buf, sum_idxs, axis=1)
    return cm_buf

def over_gt(cm):
    return (cm.T/ np.sum(cm, axis=1)).T

def main(tile_id):
    input_files = r"D:\91_classes\10_MSc\04_results\VSC\4\test20\2011_%s_c*.laz"% tile_id
    #input_files = r"D:\91_classes\10_MSc\04_results\VSC\28\test36\area1_aoi_c*_test.laz"
    #input_files = r"D:\91_classes\10_MSc\04_results\VSC\32\test33\merge.las"

    filelist = glob.glob(input_files)
    cm_sum = np.zeros((30,30), dtype=np.int64)
    pt_cnt = 0
    for idx, file in enumerate(filelist):
        print("Loading dataset '%s' (%s/%s)" % (file, idx+1, len(filelist)))
        ds = Dataset(file)
        ref = list(ds.labels)
        gt_col = ds.names.index('estim_class')
        gt = list(ds.points_and_features[:, gt_col+3])
        labels = ref #[item+2 for item in ref]
        classes = gt #[item+2 for item in gt]
        pt_cnt += len(ref)
        print("Creating confusion matrix")
        eval_cm = confusion_matrix(labels, classes, range(30))
        cm_sum += eval_cm

    keep_classes = (2,3,4,5,6,9)#(2,3,4,5) #(0, 1, 2, 3, 4, 5, 6, 7, 8)
    # confusion matrix plot
    print("Plotting")
    fig = plt.figure(figsize=(10, 10))
    num_classes = len(keep_classes) + 1
    keep_classes_e = keep_classes + (-1,)
    gs = gridspec.GridSpec(num_classes, num_classes)

    cm_sum = get_cm_compressed(cm_sum, keep_classes, delete=True)
    conf_all = over_gt(cm_sum)
    row = -1
    for ref_idx, ref_class in enumerate(keep_classes_e):
        curr_ref_axis = None
        row += 1
        col = -1
        for eval_idx, eval_class in enumerate(keep_classes_e):
            col += 1
            conf = conf_all[ref_idx, eval_idx]
            if curr_ref_axis:
                plt.subplot(gs[row, col], sharey=curr_ref_axis)
            else:
                curr_ref_axis = plt.subplot(gs[row, col])

            plt.plot([0], [0])
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            #plt.plot(points_seen, conf_timeline)

            if col == row:
                if col == num_classes-1:
                    plt.gca().set_facecolor('gray')
                    highcolor = 'k'
                    lowcolor = 'k'
                else:
                    plt.gca().set_facecolor(([30/255, 180/255, 60/255, conf]))
                    highcolor = 'xkcd:forest green'
                    lowcolor = 'xkcd:grass green'
            else:
                plt.gca().set_facecolor(([220/255, 60/255, 30/255, conf]))
                highcolor = 'xkcd:orange red'
                lowcolor = 'xkcd:dirty pink'

            plt.text(0.5,
                     0.5,
                     "%.1f%%" % (conf * 100) if not np.isnan(conf) else "N/A", ha='center',
                     )#color=highcolor if conf > 0.5 else lowcolor)
            cm = cm_sum
            ref_sum = np.sum(cm, axis=1)[ref_idx]
            eval_sum = np.sum(cm, axis=0)[eval_idx]
            plt.text(0.5,
                     0.3,
                     "%d" % (cm[ref_idx, eval_idx]), ha='center')
            if col == 0:
                plt.ylabel('%s\n%d\n(%.0f%%)' % (class_names[ref_class],
                                                 ref_sum,
                                                 ref_sum / (pt_cnt) * 100))
            if row == 0:
                plt.gca().xaxis.set_label_position('top')
                plt.xlabel('%s\n%d\n(%.0f%%)' % (class_names[eval_class],
                                                 eval_sum,
                                                 eval_sum / (pt_cnt) * 100))

            plt.gca().get_yaxis().set_ticks([])
            plt.gca().get_xaxis().set_ticks([])

            plt.ylim([0, 1])

    print("saving plot")
    fig.text(0.5, 0.94, 'Estimated', ha='center', va='center',  fontweight='bold')
    fig.text(0.06, 0.5, 'Ground truth', ha='center', va='center', rotation='vertical',  fontweight='bold')

    plt.subplots_adjust(hspace=.0, wspace=.0)
    plt.savefig((r"D:\91_classes\10_MSc\04_results\VSC\4\test20\2011_%s_cm3.png" % tile_id).replace("*", "all"))
    #plt.savefig((r"D:\91_classes\10_MSc\04_results\VSC\28\test36\conf.png"))
    #plt.savefig(r"D:\91_classes\10_MSc\04_results\VSC\32\test33\merge.png")

main('13235203')
main('13245200')
main('13205000')
main('11275100')
main('*')