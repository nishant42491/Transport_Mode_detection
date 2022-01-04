import random
from operator import itemgetter

from data_enrich import DataEnrich


class DataLoader:

    label_mapping = {
        'car': 0,
        'walk': 1,
        'bus': 2,
        'train': 3,
        'subway': 4,
        'bike': 5,
        'run': 6,
        'boat': 7,
        'airplane': 8,
        'motorcycle': 9,
        'taxi': 10
    }

    fields_to_feed = ["dist", "speed", "accel", "timedelta", "jerk", "bearing", "bearing_rate"]
    labels_to_remove = ["boat", "motorcycle", "airplane", "run", "bike"]


    def __init__(self, test_ratio=0.2, val_ratio=0.1, batchsize=4, read_from_pickle=True):
        de = DataEnrich()
        self._raw = de.get_enriched_data(read_from_pickle)
        self._test_ratio = test_ratio
        self._val_ratio = val_ratio
        self._batchsize = batchsize

    def _remove_traj_containing_labels(self):
        cleaned = []
        for elem in self._raw:
            if len(elem) == 0:
                continue
            if all(x not in list(elem["label"]) for x in self.labels_to_remove):
                cleaned.append(elem)
        self._raw = cleaned

    def _merge_labels(self, target_label, label_to_remove):
        for elem in self._raw:
            if label_to_remove in list(elem["label"]):
                elem["label"] = elem["label"].replace(to_replace=label_to_remove, value=target_label)

    def _labels_to_int_repr(self):
        for elem in self._raw:
            elem["label"] = elem["label"].apply(lambda x: self.label_mapping[x])

    def _get_split_indices(self, traj):
        train_size = int((1 - self._test_ratio) * len(traj))
        val_size = len(traj) - int((1 - self._val_ratio) * len(traj))

        indices = [x for x in range(len(traj))]

        indices_for_training = random.sample(indices, train_size)
        indices_for_validation = random.sample(indices_for_training, val_size)
        indices_for_training = set(indices_for_training) - set(indices_for_validation)
        indices_for_testing = set(indices) - indices_for_training
        indices_for_testing = list(indices_for_testing)

        return list(indices_for_training), list(indices_for_testing), list(indices_for_validation)

    def _set_splitted_data(self, traj, labels):

        i_train, i_test, i_val = self._get_split_indices(traj)

        random.shuffle(i_train)

        self.test_data = list(itemgetter(*i_test)(traj))
        self.val_data = list(itemgetter(*i_val)(traj))
        self.train_data = list(itemgetter(*i_train)(traj))
        self.test_labels = list(itemgetter(*i_test)(labels))
        self.val_labels = list(itemgetter(*i_val)(labels))
        self.train_labels = list(itemgetter(*i_train)(labels))

    def _split_too_long_traj(self, traj, labels, max_points):
        if len(traj) > max_points*2:
            splitted_traj, splitted_labels = [],[]
            num_subsets = len(traj) // max_points
            print("Splitting trajectory with length ", len(traj), "in ", num_subsets, "trajectories")
            for i in range(num_subsets):
                end_pointer = len(traj)-1 if ((i+1)*max_points)+max_points > len(traj) else (i*max_points)+max_points
                traj_subset = traj[i*max_points:end_pointer]
                labels_subset = labels[i*max_points:end_pointer]
                assert len(traj_subset) == len(labels_subset)
                splitted_traj.append(traj_subset)
                splitted_labels.append(labels_subset)
            return splitted_traj, splitted_labels
        return [traj], [labels]

    def prepare_data(self):
        trajs = []
        labels = []

        self._remove_traj_containing_labels()
        self._merge_labels("car", "taxi")
        self._labels_to_int_repr()

        for elem in self._raw:
            assert len(elem) > 0
            data_ = elem[self.fields_to_feed].values.tolist()
            label_ = elem["label"].values.tolist()
            data_, label_ = self._split_too_long_traj(data_, label_, 350)
            trajs.extend(data_)
            labels.extend(label_)

        self._set_splitted_data(trajs, labels)

    def batches(self):
        for i in range(0, len(self.train_data), self._batchsize):

            if len(self.train_data[i:i + self._batchsize]) < self._batchsize:
                break  # drop last incomplete batch

            labels_sorted = sorted(self.train_labels[i:i + self._batchsize:], key=len, reverse=True)
            train_sorted = sorted(self.train_data[i:i + self._batchsize:], key=len, reverse=True)
            for p in range(len(labels_sorted)):
                    assert len(labels_sorted[p]) == len(train_sorted[p])
            yield train_sorted, labels_sorted

    def val_batches(self):
        for i in range(0, len(self.val_data), self._batchsize):

            if len(self.val_data[i:i + self._batchsize]) < self._batchsize:
                break  # drop last incomplete batch

            labels_sorted = sorted(self.val_labels[i:i + self._batchsize:], key=len, reverse=True)
            val_sorted = sorted(self.val_data[i:i + self._batchsize:], key=len, reverse=True)
            for p in range(len(labels_sorted)):
                    assert len(labels_sorted[p]) == len(val_sorted[p])
            yield val_sorted, labels_sorted

    def test_batches(self):
        for i in range(0, len(self.test_data), self._batchsize):

            if len(self.test_data[i:i + self._batchsize]) < self._batchsize:
                break  # drop last incomplete batch

            labels_sorted = sorted(self.test_labels[i:i + self._batchsize:], key=len, reverse=True)
            test_sorted = sorted(self.test_data[i:i + self._batchsize:], key=len, reverse=True)
            for p in range(len(labels_sorted)):
                    assert len(labels_sorted[p]) == len(test_sorted[p])
            yield test_sorted, labels_sorted

    def get_train_size(self):
        return len(self.train_data)

    def get_val_size(self):
        return len(self.val_data)

    def get_test_size(self):
        return len(self.test_data)