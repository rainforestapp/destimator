==========
destimator
==========

destimator makes it easy to store trained ``scikit-learn`` estimators together with their metadata (training data, package versions, performance numbers etc.). This makes it much safer to store already-trained classifiers/regressors and allows for better reproducibility (see `this talk <https://www.youtube.com/watch?v=7KnfGDajDQw>`_ by `Alex Gaynor <https://alexgaynor.net/>`_ for some rationale).

Specifically, the ``DescribedEstimator`` class proxies most calls to the original ``Estimator`` it is wrapping, but also contains the following information:

* training and test (validation) data (``features_train``, ``labels_train``, ``features_test``, ``labels_test``)
* creation date (``created_at``)
* feature names (``feature_names``)
* performance numbers on the test set (``precision``, ``recall``, ``fscore``, ``support`` via `sklearn <http://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html>`_)
* distribution info (``distribution_info``; python distribution and versions of all installed packages)
* VCS hash (``vcs_hash``, if used inside a git repository, otherwise and empty string).

An instantiated ``DescribedEstimator`` can be easily serialized using the ``.save()`` method and deserialized using either ``.from_file()`` or ``.from_url()``. Did you ever want to store your models in S3? Now it's easy!

``DescribedEstimator`` can be used as follows:

.. code-block:: python

  import numpy as np
  from sklearn.datasets import load_iris
  from sklearn.ensemble import RandomForestClassifier
  from sklearn.cross_validation import train_test_split
  from sklearn.metrics import precision_recall_fscore_support

  from destimator import DescribedEstimator


  # get some data
  iris = load_iris()
  features = iris.data
  labels = iris.target
  features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.1)

  # train an estimator as usual (in this case a RandomForestClassifier)
  clf = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=10, random_state=0)
  clf.fit(features_train, labels_train)

  # wrap the estimator in the DescribedEstimator class giving it all the training and test (validation) data
  dclf = DescribedEstimator(
      clf,
      features_train,
      labels_train,
      features_test,
      labels_test,
      iris.feature_names
  )

Now you can use the classifier as usual:

.. code-block:: python

  print(dclf.predict(features_test))
  > [2 1 2 2 0 1 0 2 2 1 2 0 2 1 2]

and you can also access a bunch of other properties, such as the training data you supplied:

.. code-block:: python

  print(dclf.feature_names)
  > ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

  print(dclf.features_test)
  > [[ 6.3  2.8  5.1  1.5]
     [ 5.6  3.   4.5  1.5]
     [ 6.7  3.1  5.6  2.4]
     [ 6.   2.7  5.1  1.6]
     [ 4.9  3.1  1.5  0.1]
     [ 6.2  2.2  4.5  1.5]
     [ 4.7  3.2  1.6  0.2]
     [ 6.9  3.1  5.1  2.3]
     [ 7.7  2.6  6.9  2.3]
     [ 5.8  2.6  4.   1.2]
     [ 7.2  3.   5.8  1.6]
     [ 5.4  3.7  1.5  0.2]
     [ 7.2  3.2  6.   1.8]
     [ 6.3  3.3  4.7  1.6]
     [ 6.8  3.2  5.9  2.3]]

  print(dclf.labels_test)
  > [2 1 2 1 0 1 0 2 2 1 2 0 2 1 2]

the performance numbers:

.. code-block:: python

  print('precision: %s' % (dclf.precision))
  > precision: [1.0, 1.0, 0.875]

  print('recall:    %s' % (dclf.recall))
  > recall:    [1.0, 0.8, 1.0]

  print('fscore:    %s' % (dclf.fscore))
  > fscore:    [1.0, 0.888888888888889, 0.9333333333333333]

  print('support:   %s' % (dclf.support))
  > support:   [3, 5, 7]

  print('roc_auc:   %s' % (dclf.roc_auc))
  > roc_auc:   0.5

or information about the Python distribution used for training:

.. code-block:: python

  from pprint import pprint
  pprint(dclf.distribution_info)

  > {'packages': ['appnope==0.1.0',
                  'decorator==4.0.4',
                  'destimator==0.0.0.dev3',
                  'gnureadline==6.3.3',
                  'ipykernel==4.2.1',
                  'ipython-genutils==0.1.0',
                  'ipython==4.0.1',
                  'ipywidgets==4.1.1',
                  'jinja2==2.8',
                  'jsonschema==2.5.1',
                  'jupyter-client==4.1.1',
                  'jupyter-console==4.0.3',
                  'jupyter-core==4.0.6',
                  'jupyter==1.0.0',
                  'markupsafe==0.23',
                  'mistune==0.7.1',
                  'nbconvert==4.1.0',
                  'nbformat==4.0.1',
                  'notebook==4.0.6',
                  'numpy==1.10.1',
                  'path.py==8.1.2',
                  'pexpect==4.0.1',
                  'pickleshare==0.5',
                  'pip==7.1.2',
                  'ptyprocess==0.5',
                  'pygments==2.0.2',
                  'pyzmq==15.1.0',
                  'qtconsole==4.1.1',
                  'requests==2.8.1',
                  'scikit-learn==0.17',
                  'scipy==0.16.1',
                  'setuptools==18.2',
                  'simplegeneric==0.8.1',
                  'terminado==0.5',
                  'tornado==4.3',
                  'traitlets==4.0.0',
                  'wheel==0.24.0'],
     'python': '3.5.0 (default, Sep 14 2015, 02:37:27) \n'
               '[GCC 4.2.1 Compatible Apple LLVM 6.1.0 (clang-602.0.53)]'}

Finally, the object can be serialized to a `zip` file containing all the above data:

.. code-block:: python

    dclf.save('./classifiers', 'dclf')

and deserialized either from a file,

.. code-block:: python

    dclf = DescribedEstimator.from_file('./classifiers/dclf.zip')

or from a URL:

.. code-block:: python

    dclf = DescribedEstimator.from_url('http://localhost/dclf.zip')
