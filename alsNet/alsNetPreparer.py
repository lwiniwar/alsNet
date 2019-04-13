from argparse import ArgumentParser
import numpy as np
import csv
import glob
import os
import sys
import dataset

def main(in_files, density, kNN, out_folder, thinFactor):
    spacing = np.sqrt(kNN*thinFactor/(np.pi*density)) * np.sqrt(2)/2 * 0.95  # 5% MARGIN
    print("Using a spacing of %.2f m" % spacing)
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    statlist = [["Filename", "StdDev_Classes", "Ground", "Lo Veg", "Hi Veg"]]
    for file_pattern in in_files:
        print(file_pattern)
        for file in glob.glob(file_pattern):
            print("Loading file %s" % file)
            d = dataset.kNNBatchDataset(file=file, k=int(kNN*thinFactor), spacing=spacing)
            while True:
                print("Processing batch %d/%d" % (d.currIdx, d.num_batches))
                points_and_features, labels, centers = d.getBatches(batch_size=1)
                idx_to_use = np.random.choice(range(int(thinFactor*kNN)), kNN)
                names = d.names
                out_name = d.filename.replace('.la', '_c%04d.la' % d.currIdx)  # laz or las
                out_path = os.path.join(out_folder, out_name)
                if points_and_features is not None:
                    stats = dataset.ChunkedDataset.chunkStatistics(labels[0], 10)
                    rest = 1 - (stats['relative'][2] +
                                stats['relative'][3] +
                                stats['relative'][4] +
                                stats['relative'][5] +
                                stats['relative'][6] +
                                stats['relative'][9])
                    perc = [stats['relative'][2],
                            stats['relative'][3],
                            stats['relative'][4],
                            stats['relative'][5],
                            stats['relative'][6],
                            stats['relative'][9],
                            rest]
                    stddev = np.std(perc) * 100
                    list_entry = [out_name, "%.3f" % stddev, *["%.3f" % p for p in perc]]
                    statlist.append(list_entry)
                    dataset.Dataset.Save(out_path, points_and_features[0][idx_to_use], names,
                                         labels=labels[0][idx_to_use], new_classes=None)
                else:
                    break

    with open(os.path.join(out_folder, "stats.csv"), "wb") as f:
        for line in statlist:
            f.write((",".join(line) + "\n").encode('utf-8'))

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--inFiles',
                        default=[],
                        required=True,
                        help='input files (wildcard supported)',
                        action='append')
    parser.add_argument('--density', type=float, required=True, help='average point density')
    parser.add_argument('--kNN', default=200000, type=int, required=True, help='how many points per batch [default: 200000]')
    parser.add_argument('--outFolder', required=True, help='where to write output files and statistics to')
    parser.add_argument('--thinFactor', type=float, default=1., help='factor to thin out points by (2=use half of the points)')
    args = parser.parse_args()

    main(args.inFiles, args.density, args.kNN, args.outFolder, args.thinFactor)