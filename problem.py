import os
import pandas as pd
import rampwf as rw
from sklearn.model_selection import StratifiedShuffleSplit



######################################################################
import os
import numpy as np
from rampwf.utils.importing import import_module_from_source
from torchscan import crawl_module
from rampwf.score_types import BaseScoreType
from rampwf.score_types.classifier_base import ClassifierBaseScoreType
import torch.nn as nn
from sklearn.metrics import accuracy_score

class SimplifiedImageClassifier(object):
    """
    SimplifiedImageClassifier workflow.

    This workflow is used to train image classification tasks, typically when
    the dataset cannot be stored in memory. It is a simplified version
    of the `ImageClassifier` workflow where there is no batch generator
    and no image preprocessor.
    Submissions need to contain one file, which by default by is named
    image_classifier.py (it can be modified by changing
    `workflow_element_names`).
    image_classifier.py needs an `ImageClassifier` class, which implements
    `fit` and `predict_proba`, where both `fit` and `predict_proba` take
    as input an instance of `ImageLoader`.

    Parameters
    ==========

    n_classes : int
        Total number of classes.

    """

    def __init__(self, n_classes, workflow_element_names=['image_classifier']):
        self.n_classes = n_classes
        self.element_names = workflow_element_names

    def train_submission(self, module_path, folder_X_array, y_array,
                         train_is=None):
        """Train an image classifier.

        module_path : str
            module where the submission is. the folder of the module
            have to contain image_classifier.py.
        X_array : ArrayContainer vector of int
            vector of image IDs to train on
            (it is named X_array to be coherent with the current API,
             but as said here, it does not represent the data itself,
             only image IDs).
        y_array : vector of int
            vector of image labels corresponding to X_train
        train_is : vector of int
           indices from X_array to train on
        """
        folder, X_array = folder_X_array
        if train_is is None:
            train_is = slice(None, None, None)
        image_classifier = import_module_from_source(
            os.path.join(module_path, self.element_names[0] + '.py'),
            self.element_names[0],
            sanitize=True
        )
        clf = image_classifier.ImageClassifier()
        img_loader = ImageLoader(
            X_array[train_is], y_array[train_is],
            folder=folder,
            n_classes=self.n_classes)
        clf.fit(img_loader)
        return clf

    def predict_proba(self, clf, img_loader):
        batch_size = 100
        n_images = len(img_loader)
        i = 0
        y_proba = np.empty((n_images, 10))
        while i < n_images:
            indexes = range(i, min(i + batch_size, n_images))
            X = clf._load_test_minibatch(img_loader, indexes)
            i += len(indexes)
            y_proba[indexes] = nn.Softmax(dim=1)(clf.net(X)).cpu().data.numpy()
        y_proba[-1,0] = 42.0
        model_info = crawl_module(clf.net, (1, 28, 28))
        #tot_params
        y_proba[-1,1] = sum(layer['grad_params'] + layer['nograd_params'] for layer in model_info['layers'])
        #tot_flops
        y_proba[-1,2] = sum(layer['flops'] for layer in model_info['layers'])
        #tot_macs
        y_proba[-1,3] = sum(layer['macs'] for layer in model_info['layers'])
        #tot_dmas
        y_proba[-1,4] = sum(layer['dmas'] for layer in model_info['layers'])
        #tot_RAM_usage
        param_size = (model_info['overall']['param_size'] + model_info['overall']['buffer_size']) / 1024 ** 2
        overhead = model_info['overheads']['framework']['fwd'] + model_info['overheads']['cuda']['fwd']
        y_proba[-1,5] = param_size + overhead
        return y_proba

    def test_submission(self, trained_model, folder_X_array):
        """Test an image classifier.

        trained_model : tuple (function, Classifier)
            tuple of a trained model returned by `train_submission`.
        X_array : ArrayContainer of int
            vector of image IDs to test on.
            (it is named X_array to be coherent with the current API,
             but as said here, it does not represent the data itself,
             only image IDs).
        """
        folder, X_array = folder_X_array
        clf = trained_model
        test_img_loader = ImageLoader(
            X_array, None,
            folder=folder,
            n_classes=self.n_classes
        )
        y_proba = self.predict_proba(clf, test_img_loader)
        return y_proba


class Error(ClassifierBaseScoreType):
    is_lower_the_better = True
    minimum = 0.0
    maximum = 100.0

    def __init__(self, name='error [%]', precision=2):
        self.name = name
        self.precision = precision

    def __call__(self, y_true_label_index, y_pred_label_index):
        return round(100.0 - 100.0 * accuracy_score(y_true_label_index, y_pred_label_index), self.precision)

class FLOPs_metric(BaseScoreType):
    def __init__(self, precision=3):
        self.precision = precision
        self.is_lower_the_better = True
        self.minimum = 0.0
        self.maximum = np.Inf
        self.name = 'FLOPs [M]'
    def __call__(self, y_true_label_index, y_pred_label_index):
        if y_pred_label_index[-1, 0]==42.0:
            return round(y_pred_label_index[-1, 2] / 1000000, self.precision)
        else:
            return np.NAN

class MACs_metric(BaseScoreType):
    def __init__(self, precision=3):
        self.precision = precision
        self.is_lower_the_better = True
        self.minimum = 0.0
        self.maximum = np.Inf
        self.name = 'MACs [M]'
    def __call__(self, y_true_label_index, y_pred_label_index):
        if y_pred_label_index[-1, 0]==42.0:
            return round(y_pred_label_index[-1, 3] / 1000000, self.precision)
        else:
            return np.NAN

class DMAs_metric(BaseScoreType):
    def __init__(self, precision=3):
        self.precision = precision
        self.is_lower_the_better = True
        self.minimum = 0.0
        self.maximum = np.Inf
        self.name = 'DMAs [M]'
    def __call__(self, y_true_label_index, y_pred_label_index):
        if y_pred_label_index[-1, 0]==42.0:
            return round(y_pred_label_index[-1, 4] / 1000000, self.precision)
        else:
            return np.NAN

class RAM_metric(BaseScoreType):
    def __init__(self, precision=3):
        self.precision = precision
        self.is_lower_the_better = True
        self.minimum = 0.0
        self.maximum = np.Inf
        self.name = 'RAM [MB]'
    def __call__(self, y_true_label_index, y_pred_label_index):
        if y_pred_label_index[-1, 0]==42.0:
            return round(y_pred_label_index[-1, 5] / 1000000, self.precision)
        else:
            return np.NAN

class Parameters_metric(BaseScoreType):
    def __init__(self, precision=3):
        self.precision = precision
        self.is_lower_the_better = True
        self.minimum = 0.0
        self.maximum = np.Inf
        self.name = 'params [M]'
    def __call__(self, y_true_label_index, y_pred_label_index):
        if y_pred_label_index[-1, 0]==42.0:
            return round(y_pred_label_index[-1, 1]/ 1000000, self.precision)
        else:
            return np.NAN

def _image_transform(x, transforms):
    from skimage.transform import rotate
    for t in transforms:
        if t['name'] == 'rotate':
            angle = np.random.random() * (
                t['u_angle'] - t['l_angle']) + t['l_angle']
            rotate(x, angle, preserve_range=True)
    return x


class ImageLoader(object):
    """
    Load and image and optionally its label.

    In image_classifier.py, both `fit` and `predict_proba` take as input
    an instance of `ImageLoader`.
    ImageLoader is used in `fit` and `predict_proba` to either load one image
    and its corresponding label  (at training time), or one image (at test
    time).
    Images are loaded by using the method `load`.

    Parameters
    ==========

    X_array : ArrayContainer of int
        vector of image IDs to train on
         (it is named X_array to be coherent with the current API,
         but as said here, it does not represent the data itself,
         only image IDs).

    y_array : vector of int or None
        vector of image labels corresponding to `X_array`.
        At test time, it is `None`.

    folder : str
        folder where the images are

    n_classes : int
        Total number of classes.
    """

    def __init__(self, X_array, y_array, folder, n_classes):
        self.X_array = X_array
        self.y_array = y_array
        self.folder = folder
        self.n_classes = n_classes
        self.nb_examples = len(X_array)

    def load(self, index):
        """
        Load and image and optionally its label.

        Load one image and its corresponding label (at training time),
        or one image (at test time).

        Parameters
        ==========

        index : int
            Index of the image to load.
            It should in between 0 and self.nb_examples - 1

        Returns
        =======

        either a tuple `(x, y)` or `x`, where:
            - x is a numpy array of shape (height, width, nb_color_channels),
              and corresponds to the image of the requested `index`.
            - y is an integer, corresponding to the class of `x`.
        At training time, `y_array` is given, and `load` returns
        a tuple (x, y).
        At test time, `y_array` is `None`, and `load` returns `x`.
        """
        from skimage.io import imread

        if index < 0 or index >= self.nb_examples:
            raise IndexError("list index out of range")

        x = self.X_array[index]
        filename = os.path.join(self.folder, '{}'.format(x))
        x = imread(filename)
        if self.y_array is not None:
            y = self.y_array[index]
            return x, y
        else:
            return x

    def parallel_load(self, indexes, transforms=None):
        """
        Load and image and optionally its label.

        Load one image and its corresponding label (at training time),
        or one image (at test time).

        Parameters
        ==========

        index : int
            Index of the image to load.
            It should in between 0 and self.nb_examples - 1

        Returns
        =======

        either a tuple `(x, y)` or `x`, where:
            - x is a numpy array of shape (height, width, nb_color_channels),
              and corresponds to the image of the requested `index`.
            - y is an integer, corresponding to the class of `x`.
        At training time, `y_array` is given, and `load` returns
        a tuple (x, y).
        At test time, `y_array` is `None`, and `load` returns `x`.
        """
        from skimage.io import imread
        from joblib import delayed, Parallel, cpu_count

        for index in indexes:
            assert 0 <= index < self.nb_examples

        n_jobs = cpu_count()
        filenames = [
            os.path.join(self.folder, '{}'.format(self.X_array[index]))
            for index in indexes]
        xs = Parallel(n_jobs=n_jobs, backend='threading')(
            delayed(imread)(filename) for filename in filenames)

        if transforms is not None:
            from functools import partial
            transform = partial(_image_transform, transforms=transforms)
            xs = Parallel(n_jobs=n_jobs, backend='threading')(
                delayed(transform)(x) for x in xs)

        if self.y_array is not None:
            ys = [self.y_array[index] for index in indexes]
            return xs, ys
        else:
            return xs

    def __iter__(self):
        for i in range(self.nb_examples):
            yield self.load(i)

    def __len__(self):
        return self.nb_examples

####################################################################

problem_title = 'MNIST classification under minimal HW'
_target_column_name = 'class'
_prediction_label_names = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

Predictions = rw.prediction_types.make_multiclass(
    label_names=_prediction_label_names)

workflow = SimplifiedImageClassifier(
    n_classes=len(_prediction_label_names),
)

# The first score will be applied on the first Predictions
score_types = [
    Error(precision=3),
    Parameters_metric(),
    FLOPs_metric(),
    MACs_metric(),
    DMAs_metric(),
    RAM_metric(),
]


def get_cv(folder_X, y):
    _, X = folder_X
    cv = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=57)
    return cv.split(X, y)


def _read_data(path, f_name):#l'ultimo del test e' duplicato
    df = pd.read_csv(os.path.join(path, 'data', f_name))
    X = df['id'].values
    y = df['class'].values
    folder = os.path.join(path, 'data', 'imgs')
    return (folder, X), y


def get_test_data(path='.'):
    f_name = 'test.csv'
    return _read_data(path, f_name)


def get_train_data(path='.'):
    f_name = 'train.csv'
    return _read_data(path, f_name)
