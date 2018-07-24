import argparse
import glob
import logging
import numpy as np
from alsNet import AlsNetContainer
from alsNetLogger import Logger
from dataset import Dataset
import os
import webbrowser
# disable tensorflow debug information:
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s [%(levelname)s]: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--inList', help='input text file, must be csv with filename;stddev;class representativity')
    parser.add_argument('--threshold', type=float, help='upper threshold for class stddev')
    parser.add_argument('--batchSize', default=10, type=int, help='batch size for training [default: 10]')
    parser.add_argument('--dropout', default=0.5, type=float, help='probability to randomly drop a neuron ' +
                                                                   'in the last layer [default: 0.5]')
    parser.add_argument('--logDir', default='log', help='directory to write html log to [default: log]')
    parser.add_argument('--multiclass', default=True, type=bool, help='label into multiple classes ' +
                                                                      '(not only ground/nonground) [default: True]')
    parser.add_argument('--multiTrain', default=1, type=int, help='how often to feed the whole training dataset [default: 1]')
    parser.add_argument('--testList', help='list with files to test on')
    parser.add_argument('--gpuID', default=None, help='which GPU to run on (default: CPU only)')
    args = parser.parse_args()
    num_classes = 30
    num_feats = 3
    inlist = args.inList
    batch_size = int(args.batchSize)
    dropout = float(args.dropout)
    threshold = float(args.threshold)
    multitrain = args.multiTrain
    testlist = args.testList
    logdir = args.logDir
    multiclass = args.multiclass
    gpu = args.gpuID
    device = ("/gpu:%d" % int(gpu)) if gpu else "/cpu"
    logger = Logger(os.path.join(logdir, 'alsnet-log.html'), training_files=[], num_points="N/A",
                    multiclass=multiclass,
                    extra="\nThreshold for selection:\n\n{threshold}\n".format(threshold=threshold))

    alsNetInstance = AlsNetContainer('log', 0.01, logger=logger, dropout=dropout, dev=device)
    logger.container = alsNetInstance
    alsNetInstance.prepare(num_feat=num_feats, num_classes=num_classes, points_in=200000, batch_size=batch_size)

    logging.info("""Training
  _____  ___    _    ___  _  _  ___  _  _   ___ 
 |_   _|| _ \  /_\  |_ _|| \| ||_ _|| \| | / __|
   | |  |   / / _ \  | | | .` | | | | .` || (_ |
   |_|  |_|_\/_/ \_\|___||_|\_||___||_|\_| \___|
   """)
    with open(inlist, "rb") as f:
        _ = f.readline()  # remove header
        rest = f.readlines()

    datasets = []
    for line in rest:
        line = line.decode('utf-8')
        linespl = line.split(",")
        if float(linespl[1]) < threshold:
            datasets.append(os.path.join(os.path.dirname(inlist), linespl[0]))

    logger.training_files = datasets

    logging.info("Found %s suitable files" % len(datasets))

    for mult_i in range(multitrain):
        logging.info("Starting iteration %d through training dataset" % mult_i)
        seens = []
        np.random.shuffle(datasets)
        for idx, file in enumerate(datasets):
            logging.info(' - FILE %d/%d (%s) -' % (idx+1, len(datasets), file))
            train_ds = Dataset(file=file)
            paf = np.expand_dims(train_ds.points_and_features, 0)
            lab = np.expand_dims(train_ds.labels, 0)

            if idx % 1 == 0 and idx > 0:
                logging.info(" --- Testing with this batch before training...")
                alsNetInstance.test_chunk(paf[0], lab[0],
                                          os.path.join(logdir, "test_%s" % train_ds.filename),
                                          train_ds.names)

            alsNetInstance.train_batch(paf, lab)
            if mult_i == 0 and idx == 1:
                webbrowser.open(logger.outfile)
            if idx % 50 == 0:
                alsNetInstance.save_model(os.path.join(logdir, 'models', 'alsNet'))

        alsNetInstance.save_model(os.path.join(logdir, 'models', 'alsNet'))

    if testlist:
        datasets = []
        with open(testlist, "rb") as f:
            _ = f.readline()  # remove header
            rest = f.readlines()
        for line in rest:
            line = line.decode('utf-8')
            linespl = line.split(",")
            datasets.append(os.path.join(os.path.dirname(inlist), linespl[0]))

        for idx,file in enumerate(datasets):
            ds = Dataset(file=file)
            paf, lab = ds.points_and_features, ds.labels
            alsNetInstance.test_chunk(paf, lab,
                                      os.path.join(logdir, "eval_%s" % ds.filename),
                                      ds.names)
