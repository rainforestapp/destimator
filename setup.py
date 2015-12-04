from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
long_description = (
    'destimator makes it easy to store trained `scikit-learn` estimators '
    'together with their metadata (training data, package versions, '
    'performance numbers etc.). This makes it much safer to store '
    'already-trained classifiers/regressors and allows for better '
    'reproducibility.'
)

setup(
    name='destimator',
    version='0.0.2',

    description='A metadata-saving proxy for scikit-learn etimators.',
    long_description=long_description,

    # The project's main homepage.
    url='https://github.com/rainforestapp/destimator',

    # Author details
    author='Maciej Gryka',
    author_email='maciej@rainforstqa.com',

    # Choose your license
    license='MIT',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: MIT License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
    ],

    # What does your project relate to?
    keywords='scikit-learn machine-learning data science',

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),

    install_requires=[
        'requests>=2.8.1',
        'numpy>=1.10.1',
        'scipy>=0.16.1',
        'scikit-learn>=0.17',
    ],

    # List additional groups of dependencies here (e.g. development
    # dependencies). You can install these using the following syntax,
    # for example:
    # $ pip install -e .[dev,test]
    extras_require={
        'dev': [
            'twine==1.6.4',
        ],
        'test': [
            'pytest',
            'flake8',
            'pytest-cov',
            'tox',
        ],
    },

    # # To provide executable scripts, use entry points in preference to the
    # # "scripts" keyword. Entry points provide cross-platform support and allow
    # # pip to create the appropriate form of executable for the target platform.
    # entry_points={
    #     'console_scripts': [
    #         'sample=sample:main',
    #     ],
    # },
)
