import os
from setuptools import setup, find_packages

NAME = "tk_nn_classifier"
VERSION = os.environ.get("TK_NN_CLASSIFIER", '0.0.0')

INSTALL_REQUIRES = [
    "tensorflow == 1.14.0",
    "numpy == 1.16.0",
    "xml-miner == 0.0.5",
    "spacy == 2.3.0",
    "en_core_web_sm == 2.3.0",
    "easy_tokenizer == 0.0.10"
]

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest', ]

setup(
    name=NAME,
    version=VERSION,
    keywords='neural network, classifier, easy config',
    description='''a general framework to easily setup a classifier to train and predict''',
    author="Chao Li",
    author_email="chaoli.job@google.com",
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    entry_points={
        "console_scripts": [
            "tk-nn-classifier=tk_nn_classifier.__main__:main",
        ],
    },
    dependency_links = [
        "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.0/en_core_web_sm-2.3.0.tar.gz#egg=en_core_web_sm",

    ],
    test_suite="tests",
    setup_requires=setup_requirements,
    tests_require=test_requirements,
    install_requires=INSTALL_REQUIRES,
    packages=find_packages(),
    zip_safe=False
)
