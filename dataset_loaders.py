
# Hands in the Million 2017 challenge dataset

HIM2017_default_dataset_root = r"/mnt/nasbi/shared/research/hand-pose-estimation/hands2017/data/hand2017_nor_img_new"
HIM2017_default_train_and_validate_list = ["nor_%08d.pkl" % i for i in range(1000, 957001, 1000)] + ["nor_00957032.pkl"]

HIM2017_default_testset_root = r"/mnt/nasbi/shared/research/hand-pose-estimation/hands2017/data/hand2017_test_0914"
HIM2017_default_test_list = ["%08d.pkl" % i for i in range(10000, 290001, 10000)] + ["00295510.pkl"]


class HIM2017Loader:
    def __init__(self,
                 dataset_root=HIM2017_default_dataset_root,
                 train_and_validate_list=HIM2017_default_train_and_validate_list,
                 train_validate_split=1.0,
                 testset_root=HIM2017_default_testset_root,
                 test_list=HIM2017_default_test_list):
        self._dataset_root = dataset_root
        self._train_and_validate_list = train_and_validate_list
        self._train_validate_split = train_validate_split
        self._train_list, self._validate_list = self.split_list(self._train_and_validate_list,
                                                                self._train_validate_split)
        self._testset_root = testset_root
        self._test_list = test_list

    def set_split(self, train_validate_split):
        self._train_validate_split = train_validate_split
        self._train_list, self._validate_list = self.split_list(self._train_and_validate_list,
                                                                self._train_validate_split)

    @property
    def train_list(self):
        return self._train_list

    @property
    def validate_list(self):
        return self._validate_list

    @property
    def test_list(self):
        return self._test_list

    @property
    def train_root(self):
        return self._dataset_root

    @property
    def validate_root(self):
        return self._dataset_root

    @property
    def test_root(self):
        return self._testset_root

    @staticmethod
    def split_list(train_validate_list, split=0.8):
        cut = int(len(train_validate_list) * split)
        return train_validate_list[:cut], train_validate_list[cut:]
