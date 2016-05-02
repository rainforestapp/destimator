from __future__ import print_function, unicode_literals

import os
import sys
import shutil
import zipfile
import datetime
import tempfile
import json as json
try:
    from cStringIO import StringIO
except ImportError:
    from io import StringIO

from contextlib import closing

import requests
import numpy as np
from sklearn.externals import joblib
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, log_loss

from .utils import get_current_vcs_hash, get_installed_packages


METADATA_VERSION = '2'


class DescribedEstimator(object):
    def __init__(
        self,
        clf,
        features_train, labels_train, features_test, labels_test, feature_names=None,
        *args, **kwargs
    ):
        """
        A trained classifier, together with all the data and metadata. After it
        is created, this object acts as a proxy to sklearn.base.BaseEstimator,
        meaning it forwards all calls (including predict and predict_proba) to
        the actual estimator. It also contains all the information necessary to
        recreate the classifier.

        Normally this object is initialized using:
            >> DescribedEstimator(clf, features_train, labels_train, features_test, labels_test, feature_names)

        If `compute_metadata` is set to False, a `metadata` argument must be
        given and `feature_names` is ignored.

        """
        self._clf = clf
        self._data = {
            'features_train': features_train,
            'labels_train': labels_train,
            'features_test': features_test,
            'labels_test': labels_test,
        }
        compute_metadata = 'metadata' not in kwargs
        if compute_metadata:
            precision, recall, fscore, support = precision_recall_fscore_support(
                labels_test,
                clf.predict(features_test)
            )

            roc_auc = roc_auc_score(labels_test, clf.predict_proba(features_test)[:, 1])
            self.metadata = {
                'metadata_version': METADATA_VERSION,
                'created_at': datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'),
                'feature_names': feature_names,
                'vcs_hash': get_current_vcs_hash(),
                'distribution_info': {
                    'python': sys.version,
                    'packages': get_installed_packages(),
                },
                'performance_scores': {
                    'precision': [float(p) for p in precision],  # cast to make it
                    'recall':    [float(r) for r in recall],     # JSONserializable
                    'fscore':    [float(f) for f in fscore],
                    'support':   [int(s) for s in support],
                    'roc_auc':   float(roc_auc),
                    'log_loss':  float(log_loss(labels_test, clf.predict_proba(features_test)[:, 1])),
                }
            }
        else:
            self.metadata = kwargs.pop('metadata')
        self.validate_data()
        super(DescribedEstimator, self).__init__(*args, **kwargs)

    def __eq__(self, other):
        data_equal = all([
            np.array_equal(self._data[k], other._data[k])
            for k in self._data.keys()
        ])
        metadata_equal = all([
            self.metadata['metadata_version'] == other.metadata['metadata_version'],
            self.metadata['feature_names'] == other.metadata['feature_names'],
            self.metadata['performance_scores'] == other.metadata['performance_scores'],
        ])
        classifier_equal = (
            self._clf.__class__ == other._clf.__class__ and
            self._clf.get_params() == other._clf.get_params()
        )
        return data_equal and metadata_equal and classifier_equal

    @classmethod
    def from_file(cls, f):
        """
        Read the described classifier from file. `f` can be a path or a
        file-like object.

        """
        zf = zipfile.ZipFile(f)
        extract_dir = tempfile.mkdtemp()
        try:
            zf.extractall(extract_dir)
            data = {}
            for fn in zf.namelist():
                fn_full = os.path.join(extract_dir, fn)
                if fn == 'model.bin':
                    data['clf'] = joblib.load(fn_full)
                elif fn == 'features_train.bin':
                    data['features_train'] = joblib.load(fn_full)
                elif fn == 'labels_train.bin':
                    data['labels_train'] = joblib.load(fn_full)
                elif fn == 'features_test.bin':
                    data['features_test'] = joblib.load(fn_full)
                elif fn == 'labels_test.bin':
                    data['labels_test'] = joblib.load(fn_full)
                elif fn == 'metadata.json':
                    data['metadata'] = json.loads(open(fn_full, 'rt').read())
            return cls(
                clf=data['clf'],
                features_train=data['features_train'],
                labels_train=data['labels_train'],
                features_test=data['features_test'],
                labels_test=data['labels_test'],
                metadata=data['metadata'],
            )
        finally:
            shutil.rmtree(extract_dir)

    @classmethod
    def from_url(cls, url):
        """Read the described classifier from the given URL."""
        f = StringIO()
        f.write(requests.get(url, stream=True).content)
        return cls.from_file(f)

    def is_compatible(self, other):
        """
        Check whether this estimator is compatible with the other one (in other
        words, are their `feature_names` the same).

        """
        return self.feature_names == other.feature_names

    @property
    def n_training_samples_(self):
        return self._data['features_train'].shape[0]

    @property
    def n_features_(self):
        if hasattr(self._clf, 'n_features_'):
            return self._clf.n_features_
        else:
            return self._data['features_train'].shape[1]

    def save(self, path, filename=None):
        """Save the classifier under `path`."""
        if filename is None:
            filename = 'destimator_' + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        filename += '.zip'
        try:
            os.makedirs(path)
        except OSError:
            pass  # the directory exists
        archive_name = os.path.join(path, filename)
        # save files to disk (unfortunately joblib requires that)
        joblib.dump(self._clf, os.path.join(path, 'model.bin'), compress=9)
        joblib.dump(self.features_train, os.path.join(path, 'features_train.bin'), compress=9)
        joblib.dump(self.labels_train, os.path.join(path, 'labels_train.bin'), compress=9)
        joblib.dump(self.features_test, os.path.join(path, 'features_test.bin'), compress=9)
        joblib.dump(self.labels_test, os.path.join(path, 'labels_test.bin'), compress=9)
        with open(os.path.join(path, 'metadata.json'), 'wt') as f:
            f.write(json.dumps(self.metadata))
        # pack all the files into a zip archive
        with closing(zipfile.ZipFile(archive_name, 'w', zipfile.ZIP_DEFLATED)) as zf:
            for fn in ['model.bin', 'features_train.bin', 'labels_train.bin', 'features_test.bin', 'labels_test.bin', 'metadata.json']:
                zf.write(os.path.join(path, fn), fn)
        return archive_name

    def validate_data(self):
        if len(self._data['features_train']) != len(self._data['labels_train']):
            raise ValueError('Number of features and labels must be the same.')

        features_mismatch = self.features_train.shape[1] != self.features_test.shape[1]
        labels_dims_train = self.labels_train.shape[1] if self.labels_train.ndim > 1 else 1
        labels_dims_test = self.labels_test.shape[1] if self.labels_test.ndim > 1 else 1
        labels_mismatch = labels_dims_train != labels_dims_test
        if features_mismatch or labels_mismatch:
            raise ValueError('Training and test data must have the same dimensionality.')

        if self.feature_names is not None and self.features_train.shape[1] != len(self.feature_names):
            raise ValueError('Number of feature_names must match the number of features.')

        if 'feature_names' not in self.metadata:
            raise ValueError('feature_names mist be provided.')

    def __getattr__(self, name):
        if name in self._data.keys():
            return self._data[name]
        elif name in self.metadata.keys():
            return self.metadata[name]
        elif name in self.metadata['performance_scores'].keys():
            return self.metadata['performance_scores'][name]
        else:
            return getattr(self._clf, name)
