import datetime
import numpy as np


class AlsNetHistory:
    def __init__(self):
        self.cm = []
        self.points_seen = []
        self.timestamps = []
        self.losses = []

    def add_history_step(self, cm, points_seen, loss, timestamp=datetime.datetime.now()):
        self.cm.append(cm+1e-8)  # +1e-8 to make sure always > 0
        self.points_seen.append(points_seen)
        self.losses.append(loss)
        self.timestamps.append(timestamp)

    def get_cm_timeline(self, i, j):
        return [AlsNetHistory.over_gt(cm)[i, j] for cm in self.cm]

    def get_cm_timeline_compressed(self, i, j, keep_classes):
        return [AlsNetHistory.over_gt(AlsNetHistory.get_cm_compressed(cm, keep_classes))[i, j] for cm in self.cm]

    def get_oa_timeline(self):
        return [np.sum([cm[i, i] for i in range(cm.shape[0])]) / np.sum(cm, axis=(0,1)) for cm in self.cm]

    def get_oa_timeline_smooth(self, n_window):
        return np.convolve(self.get_oa_timeline(), np.ones((n_window, ))/n_window, mode='valid')

    @staticmethod
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

    @staticmethod
    def over_gt(cm):
        return (cm.T/ np.sum(cm, axis=1)).T

    def class_points_timeline(self, class_idx):
        return [np.sum(cm[class_idx, :]) for cm in self.cm]


if __name__ == '__main__':
    cm = np.array([[45, 3, 4, 6, 3],
                   [2, 18, 3, 5, 4],
                   [9, 1, 13, 5, 7],
                   [0, 4, 3, 15, 3],
                   [2, 8, 3, 5, 14]])
    cm_c = AlsNetHistory.get_cm_compressed(cm, (0, 1))
    print(cm)
    print(cm_c)
