from __future__ import print_function, unicode_literals

import os
import shutil
import zipfile
import datetime
import tempfile
import subprocess

import pytest
import numpy as np
from numpy.testing import assert_almost_equal
from sklearn.dummy import DummyClassifier

from destimator import DescribedEstimator, utils


@pytest.fixture
def features():
    return np.zeros([10, 3])


@pytest.fixture
def labels():
    labels = np.zeros(10)
    labels[5:] = 1.0
    return labels


@pytest.fixture
def clf(features, labels):
    clf = DummyClassifier(strategy='constant', constant=0.0)
    clf.fit(features, labels)
    return clf


@pytest.fixture
def clf_described(clf, features, labels, feature_names):
    return DescribedEstimator(clf, features, labels, features, labels, feature_names)


@pytest.fixture
def feature_names():
    return ['one', 'two', 'three']


class TestDescribedEstimator(object):
    def test_init(self, clf_described):
        assert clf_described.n_training_samples_ == 10
        assert clf_described.n_features_ == 3

    def test_init_error(self, clf, features, labels, feature_names):
        with pytest.raises(ValueError):
            wrong_labels = np.zeros([9, 1])
            DescribedEstimator(clf, features, wrong_labels, features, labels, feature_names)

        with pytest.raises(ValueError):
            wrong_feature_names = ['']
            DescribedEstimator(clf, features, labels, features, labels, wrong_feature_names)

    def test_from_file(self, clf_described):
        save_dir = tempfile.mkdtemp()
        try:
            file_path = clf_described.save(save_dir)
            destimator = DescribedEstimator.from_file(file_path)
            assert destimator == clf_described
        finally:
            shutil.rmtree(save_dir)

    def test_is_compatible(self, clf, clf_described, features, labels):
        compatible = DescribedEstimator(clf, features, labels, features, labels, ['one', 'two', 'three'])
        assert clf_described.is_compatible(compatible)

        incompatible = DescribedEstimator(clf, features, labels, features, labels, ['one', 'two', 'boom'])
        assert not clf_described.is_compatible(incompatible)

    def test_metadata(self, clf, features, labels, feature_names):
        clf_described = DescribedEstimator(clf, features, labels, features, labels, feature_names)
        d = clf_described.metadata
        assert d['feature_names'] == feature_names
        # assert type(d['metadata_version']) == str
        assert type(datetime.datetime.strptime(d['created_at'], '%Y-%m-%d-%H-%M-%S')) == datetime.datetime
        # assert type(d['vcs_hash']) == str
        assert type(d['distribution_info']) == dict
        # assert type(d['distribution_info']['python']) == str
        assert type(d['distribution_info']['packages']) == list
        assert type(d['performance_scores']['precision']) == list
        assert type(d['performance_scores']['precision'][0]) == float
        assert type(d['performance_scores']['recall']) == list
        assert type(d['performance_scores']['recall'][0]) == float
        assert type(d['performance_scores']['fscore']) == list
        assert type(d['performance_scores']['fscore'][0]) == float
        assert type(d['performance_scores']['support']) == list
        assert type(d['performance_scores']['support'][0]) == int
        assert type(d['performance_scores']['roc_auc']) == float
        assert type(d['performance_scores']['log_loss']) == float

    def test_get_metric(self, clf_described):
        assert clf_described.recall == [1.0, 0.0]
        assert clf_described.roc_auc == 0.5
        # log_loss use epsilon 1e-15, so -log(1e-15) / 2 approximately equal 20
        assert_almost_equal(clf_described.log_loss, 17.269, decimal=3)

    def test_save_classifier(self, clf_described):
        save_dir = tempfile.mkdtemp()
        try:
            saved_name = clf_described.save(save_dir)
            assert os.path.dirname(saved_name) == save_dir
            assert os.path.isfile(saved_name)
            assert saved_name.endswith('.zip')
            zf = zipfile.ZipFile(saved_name)
            files_present = zf.namelist()
            expected_files = [
                'model.bin', 'features_train.bin', 'labels_train.bin',
                'features_test.bin', 'labels_test.bin', 'metadata.json',
            ]
            # could use a set, but this way errors are easier to read
            for f in expected_files:
                assert f in files_present
        finally:
            shutil.rmtree(save_dir)

    def test_save_classifier_with_filename(self, clf_described):
        save_dir = tempfile.mkdtemp()
        try:
            saved_name = clf_described.save(save_dir, filename='boom.pkl')
            assert os.path.basename(saved_name) == 'boom.pkl.zip'
            assert os.path.isfile(saved_name)
        finally:
            shutil.rmtree(save_dir)

    def test_save_classifier_nonexistent_path(self, clf_described):
        save_dir = tempfile.mkdtemp()
        try:
            saved_name = clf_described.save(os.path.join(save_dir, 'nope'))
            os.path.dirname(saved_name) == save_dir
            assert os.path.isfile(saved_name)
        finally:
            shutil.rmtree(save_dir)


class TestGetCurrentGitHash(object):
    def test_get_current_vcs_hash(self, monkeypatch):
        def fake_check_output(*args, **kwargs):
            return b'thisisagithash'
        monkeypatch.setattr(subprocess, 'check_output', fake_check_output)
        assert utils.get_current_vcs_hash() == 'thisisagithash'

    def test_get_current_vcs_hash_no_git(self, monkeypatch):
        def fake_check_output(*args, **kwargs):
            raise OSError()
        monkeypatch.setattr(subprocess, 'check_output', fake_check_output)
        assert utils.get_current_vcs_hash() == ''

    def test_get_current_vcs_hash_git_error(self, monkeypatch):
        def fake_check_output(*args, **kwargs):
            raise subprocess.CalledProcessError(0, '', '')
        monkeypatch.setattr(subprocess, 'check_output', fake_check_output)
        assert utils.get_current_vcs_hash() == ''
