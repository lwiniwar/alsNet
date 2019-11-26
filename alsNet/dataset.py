import numpy as np
import laspy
import os
from scipy.spatial import KDTree
from sklearn.preprocessing import normalize
import logging


class Dataset():
    ATTR_EXLUSION_LIST = ['X', 'Y', 'Z', 'raw_classification', 'Classification',
                          'flag_byte', 'scan_angle_rank', 'user_data',
                          'pt_src_id', 'gps_time']
    ATTR_EXTRA_LIST = ['num_returns', 'return_num']

    def __init__(self, file, load=True, multiclass=True, normalize=False):
        self.file = file
        self._features = self._xyz = self._classes = self._names = None
        self.xmax = self.xmin = self.ymax = self.ymin = None
        self._header = None
        self.multiclass = multiclass
        self.normalize = normalize
        if load:
            self.load_data()

    def load_data(self):
        file_h = laspy.file.File(self.file, mode='r')
        self._xyz = np.vstack([file_h.x, file_h.y, file_h.z]).transpose()
        self._classes = file_h.classification
        points = file_h.points['point']
        attr_names = [a for a in points.dtype.names] + Dataset.ATTR_EXTRA_LIST
        self._features = np.array([getattr(file_h, name) for name in attr_names
                                   if name not in Dataset.ATTR_EXLUSION_LIST]).transpose()
        self._names = [name for name in attr_names if name not in Dataset.ATTR_EXLUSION_LIST]

        self.xmin = file_h.header.min[0]
        self.ymin = file_h.header.min[1]
        self.xmax = file_h.header.max[0]
        self.ymax = file_h.header.max[1]
        self._header = file_h.header
        file_h.close()

    def statistics(self):
        stats = {'absolute': {},
                 'relative': {}}
        for i in range(np.max(self.labels)):
            count = np.count_nonzero(self.labels == i)
            stats['absolute'][i] = count
            stats['relative'][i] = count/len(self)

        return stats

    @property
    def labels(self):
        if self._xyz is None:
            self.load_data()
        ret_val = self._classes if self.multiclass else (self._classes != 2).astype('int8') + 2
        return ret_val

    @property
    def names(self):
        return self._names

    @property
    def points_and_features(self):
        if self._xyz is None:
            self.load_data()
        ret_val = np.hstack((self._xyz, self._features))
        if self.normalize:
            normalize(ret_val)
        return ret_val

    @property
    def filename(self):
        return os.path.basename(self.file)

    def points_and_features_f(self):
        return self.points_and_features

    def labels_f(self):
        return self.labels

    def unload(self):
        self._features = self._xyz = self._classes = self._names = None
        self.xmax = self.xmin = self.ymax = self.ymin = None
        self._header = None

    def get_label_unique_count(self):
        return len(np.unique(self._classes))

    def get_feature_count(self):
        return self._features.shape[1]


    def __len__(self):
        return self.labels.shape[0]

    def getBatch(self, start_idx, batch_size, idx_randomizer=None):
        if idx_randomizer is not None:
            idx_range = idx_randomizer[start_idx:start_idx + batch_size]
        else:
            idx_range = range(start_idx, start_idx + batch_size)
        data = self.points_and_features[idx_range]
        labels = self.labels[idx_range]

    def save_with_new_classes(self, outFile, new_classes):
        inFile = laspy.file.File(self.file)
        outFile = laspy.file.File(outFile, mode='w', header=inFile.header)
        outFile.points = inFile.points
        outFile.Classification = new_classes[0]
        outFile.close()

    @staticmethod
    def Save(path, points_and_features, names=None, labels=None, new_classes=None, probs=None):
        hdr = laspy.header.Header()
        outfile = laspy.file.File(path, mode="w", header=hdr)
        if new_classes is not None:
            outfile.define_new_dimension(name="estim_class", data_type=5, description="estimated class")
        if labels is not None and new_classes is not None:
            outfile.define_new_dimension(name="class_correct", data_type=5, description="correctness of estimated class")
        if probs is not None:
            for classid in range(probs.shape[1]):
                outfile.define_new_dimension(name="prob_class%02d" % classid, data_type=9, description="p of estimated class %02d"%classid)

        allx = points_and_features[:, 0]
        ally = points_and_features[:, 1]
        allz = points_and_features[:, 2]

        xmin = np.floor(np.min(allx))
        ymin = np.floor(np.min(ally))
        zmin = np.floor(np.min(allz))

        outfile.header.offset = [xmin, ymin, zmin]
        outfile.header.scale = [0.001, 0.001, 0.001]

        outfile.x = allx
        outfile.y = ally
        outfile.z = allz

        for featid in range(points_and_features.shape[1]-3):
            try:
                data = points_and_features[:, 3+featid]
                if names[featid] in ['num_returns', 'return_num']:  # hack to treat int-values
                    data = data.astype('int8')
                setattr(outfile, names[featid], data)
            except Exception as e:
                logging.warning("Could not save attribute %s to file %s: \n%s" % (names[featid], path, e))
                #raise

        if probs is not None:
            for classid in range(probs.shape[1]):
                setattr(outfile, "prob_class%02d" % classid, probs[:, classid])

        if labels is not None:
            outfile.classification = labels
        if new_classes is not None:
            outfile.estim_class = new_classes
        if labels is not None and new_classes is not None:
            outfile.class_correct = np.equal(labels, new_classes)*-1 + 6  #  so that equal =5 --> green (veg)
            #  and not equal =6 --> red (building)

        outfile.close()


class ChunkedDataset(Dataset):
    def __init__(self, chunk_size, overlap, *args, **kwargs):
        super(ChunkedDataset, self).__init__(*args, **kwargs)
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.curr_chunk = 0

        self.num_cols = (self.xmax - self.xmin) // (self.chunk_size - self.overlap) + 1
        self.num_rows = (self.ymax - self.ymin) // (self.chunk_size - self.overlap) + 1

    def idx_to_lims(self, idx):
        if idx >= self.num_cols * self.num_rows:
            return None
        row_idx = idx // self.num_cols
        col_idx = idx % self.num_cols

        return [self.xmin + (self.chunk_size - self.overlap) * col_idx,
                self.xmin + (self.chunk_size - self.overlap) * (col_idx + 1) + self.overlap,
                self.ymin + (self.chunk_size - self.overlap) * row_idx,
                self.ymin + (self.chunk_size - self.overlap) * (row_idx + 1) + self.overlap,
                ]

    def getNextChunk(self):
        lims = self.idx_to_lims(self.curr_chunk)
        if not lims:  # no more chunks
            return None, None
        idxes = self._xyz[:, 0] >= lims[0]
        idxes &= self._xyz[:, 0] < lims[1]
        idxes &= self._xyz[:, 1] >= lims[2]
        idxes &= self._xyz[:, 1] < lims[3]
        self.curr_chunk += 1
        return self.points_and_features[idxes, :], self.labels[idxes]

    @staticmethod
    def chunkStatistics(labels, max):
        stats = {'absolute': {},
                 'relative': {}}
        for i in range(max):
            count = np.count_nonzero(labels == i)
            stats['absolute'][i] = count
            stats['relative'][i] = count / len(labels)

        return stats

class kNNBatchDataset(Dataset):

    def __init__(self, k, spacing, *args, **kwargs):
        super(kNNBatchDataset, self).__init__(*args, **kwargs)
        self.spacing = spacing
        self.k = k
        self.tree = None
        self.currIdx = 0

        self.num_cols = (self.xmax - self.xmin - self.spacing/2) // (self.spacing) + 1
        self.num_rows = (self.ymax - self.ymin - self.spacing/2) // (self.spacing) + 1
        self.num_batches = int(self.num_cols * self.num_rows)
        self.rndzer = list(range(self.num_batches))
        np.random.shuffle(self.rndzer)
        self.buildKD()

    def buildKD(self):
        logging.info(" -- Building kD-Tree with %d points..." % len(self))
        self.tree = KDTree(self._xyz[:, :2], leafsize=100)  # build only on x/y
        logging.info(" --- kD-Tree built.")


    def getBatches(self, batch_size=1):
        centers = []
        for i in range(batch_size):
            if self.currIdx >= self.num_batches:
                break
            centers.append([self.xmin + self.spacing/2 + (self.currIdx // self.num_rows) * self.spacing,
                            self.ymin + self.spacing/2 + (self.currIdx % self.num_rows) * self.spacing])
            self.currIdx += 1
        if centers:
            _, idx = self.tree.query(centers, k=self.k)
            return self.points_and_features[idx, :], self.labels[idx]
        else:
            return None, None

    def getBatchByIdx(self, batch_idx):
        centers = [[self.xmin + self.spacing / 2 + (batch_idx // self.num_rows) * self.spacing,
                    self.ymin + self.spacing / 2 + (batch_idx % self.num_rows) * self.spacing]]
        _, idx = self.tree.query(centers, k=self.k)
        return self.points_and_features[idx, :], self.labels[idx]
