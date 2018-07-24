import argparse
import glob
import logging
import numpy as np
from alsNet import AlsNetContainer
from alsNetLogger import Logger
from dataset import ChunkedDataset
from dataset import kNNBatchDataset
import os
# disable tensorflow debug information:
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s [%(levelname)s]: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--inFiles', default='./*.laz', help='input files (wildcard supported) [default: ./*.laz]')
    parser.add_argument('--testFiles', type=str, help='files for testing/validation ')
    parser.add_argument('--batchSize', default=10, type=int, help='batch size for training [default: 10]')
    parser.add_argument('--kNN', default=100000, type=int, help='how many points per batch [default: 100000]')
    parser.add_argument('--spacing', default=100, type=float, help='spatial spacing between batches in m [default: 100]')
    parser.add_argument('--dropout', default=0.5, type=float, help='probability to randomly drop a neuron ' +
                                                                   'in the last layer [default: 0.5]')
    parser.add_argument('--logDir', default='log', help='directory to write html log to [default: log]')
    parser.add_argument('--multiclass', default=True, type=bool, help='label into multiple classes ' +
                                                                      '(not only ground/nonground) [default: True]')
    args = parser.parse_args()
    num_classes = 30
    batch_size = int(args.batchSize)
    train_files = glob.glob(args.inFiles)
    test_files = glob.glob(args.testFiles)
    dropout = float(args.dropout)
    kNN = int(args.kNN)
    spacing = int(args.spacing)
    logdir = args.logDir
    multiclass = args.multiclass
    logger = Logger(os.path.join(logdir, 'alsnet-log.html'), training_files=train_files, num_points=kNN,
                    multiclass=multiclass)
    alsNetInstance = AlsNetContainer('log', 0.01, logger=logger, dropout=dropout)
    logger.container = alsNetInstance
    alsNetInstance.prepare(num_feat=1, num_classes=num_classes, points_in=kNN, batch_size=batch_size)

    logging.info("""Training
  _____  ___    _    ___  _  _  ___  _  _   ___ 
 |_   _|| _ \  /_\  |_ _|| \| ||_ _|| \| | / __|
   | |  |   / / _ \  | | | .` | | | | .` || (_ |
   |_|  |_|_\/_/ \_\|___||_|\_||___||_|\_| \___|
   """)


    for idx, file in enumerate(train_files):
        logging.info(' - FILE %d/%d (%s) -' % (idx+1, len(train_files), file))
        train_ds = kNNBatchDataset(file=file, k=kNN, spacing=spacing, multiclass=multiclass)
        batch_idx = 0
        test_paf, test_labels = train_ds.getBatchByIdx(1200)
        while True:
            logging.info(" -- Fetching batches (%d-%d)/%d..." % (batch_idx + 1,
                                                                 batch_size + batch_idx - 1 + 1,
                                                                 train_ds.num_batches))
            points_and_features, labels = train_ds.getBatches(batch_size=batch_size)
            batch_idx += 1
            #if labels is not None and np.max(labels) > num_classes:
            #    logging.warning("Chunk contains points with Classes: %s" % str(np.unique(labels)))
            #    logging.warning("but only classes 0-%s are defined in the model." % num_classes)
            #    logging.warning("Removing those points...")
            #    points_and_features = np.delete(points_and_features, labels > num_classes, axis=0)
            #    labels = np.delete(labels, labels > num_classes, axis=0)
            if points_and_features is not None:
                logging.info(" -- Feeding batches (%d-%d)/%d..." % (batch_idx, batch_size+batch_idx-1, train_ds.num_batches))
                #logging.info("Chunk %d/%d (%d points)" % (train_ds.curr_chunk, train_ds.num_rows * train_ds.num_cols, points_and_features.shape[0]))
                stats = ChunkedDataset.chunkStatistics(labels[0], num_classes)
                #logging.info("Stats: %5.2f Ground | %5.2f Building | %5.2f Hi Veg | %5.2f Med Veg | %5.2f Lo Veg" %
                #             (stats['relative'][2]*100, stats['relative'][6]*100, stats['relative'][5]*100,
                #              stats['relative'][4]*100, stats['relative'][3]*100))
                logger.perc_building.append(stats['relative'][6]*100)
                logger.perc_hi_veg.append(stats['relative'][5]*100)
                logger.perc_med_veg.append(stats['relative'][4]*100)
                logger.perc_lo_veg.append(stats['relative'][3]*100)
                logger.perc_ground.append(stats['relative'][2]*100)
                logger.perc_water.append(stats['relative'][9]*100)
                rest = 1 - (stats['relative'][2] +
                            stats['relative'][3] +
                            stats['relative'][4] +
                            stats['relative'][5] +
                            stats['relative'][6] +
                            stats['relative'][9])
                logger.perc_rest.append(rest*100)
                perc = [stats['relative'][2],
                        stats['relative'][3],
                        stats['relative'][4],
                        stats['relative'][5],
                        stats['relative'][6],
                        stats['relative'][9],
                        rest]
                stddev = np.std(perc) * 100
                logger.losses.append(stddev)
                if len(logger.points_seen) == 0:
                    logger.points_seen.append(0)
                else:
                    logger.points_seen.append(logger.points_seen[-1] + kNN*1e-6)
                logger.accuracy_train.append(0)
                logger.cumaccuracy_train.append(0)
                logger.save()
                if batch_idx % 10 == 0:
                    logging.info(" --- Testing with this batch before training...")
                    alsNetInstance.test_chunk(points_and_features[0], labels[0],
                                              os.path.join(logdir, "test_batch_%s_%s" % (batch_idx, train_ds.filename)))
                    logging.info(" --- Testing with fixed 'batch of interest'...")
                    alsNetInstance.test_chunk(test_paf[0], test_labels[0],
                                              os.path.join(logdir, "const_test_%s_%s" % (batch_idx, train_ds.filename)))
                alsNetInstance.train_batch(points_and_features, labels)
                alsNetInstance.save_model('/data/lwiniwar/02_temp/models/alsNet')
            else:
                break

    logging.info("""Testing
  _____  ___  ___  _____  ___  _  _   ___ 
 |_   _|| __|/ __||_   _||_ _|| \| | / __|
   | |  | _| \__ \  | |   | | | .` || (_ |
   |_|  |___||___/  |_|  |___||_|\_| \___|
   """)
    for idx, file in enumerate(test_files):
        logging.info(' - FILE %d/%d (%s) - ' % (idx, len(test_files), file))
        test_ds = kNNBatchDataset(file=file, k=kNN, spacing=spacing, multiclass=multiclass)
        while True:
            points_and_features, labels = test_ds.getBatches(batch_size=1)
            if points_and_features is not None:
                logging.info(" -- Evaluating region %d/%d" % (test_ds.currIdx, test_ds.num_batches))
                stats = ChunkedDataset.chunkStatistics(labels[0], num_classes)
                logging.info(" -- Stats: %5.2f Ground | %5.2f Building | %5.2f Hi Veg | %5.2f Med Veg | %5.2f Lo Veg" %
                             (stats['relative'][2]*100, stats['relative'][6]*100, stats['relative'][5]*100,
                              stats['relative'][4]*100, stats['relative'][3]*100))
                alsNetInstance.test_chunk(points_and_features[0], labels[0], file.replace(".laz", "eval/region_%d.laz" % test_ds.currIdx))
            else:
                break