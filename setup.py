import os
from setuptools import setup, find_packages

NAME = "recruitment_agency_detector"
VERSION = os.environ.get("RECRUITMENT_AGENCY_DETECTOR", '0.0.0')

INSTALL_REQUIRES = [
    "tensorflow>=1.13.1",
    "numpy>=1.16.0",
    "spacy>=2.1.8",
    "xml-miner>=0.0.3",
    "en-core-web-sm>=2.1.0",
    "tk-preprocessing>=0.0.4"
]

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest', ]

setup(
    name=NAME,
    version=VERSION,
    keywords='recruitment agency, vacancy, detector',
    description='''a detector to detect whether vacancy is from a recruitment
    agency''',
    author="Chao Li",
    author_email="chao@textkernel.com",
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    scripts=[
        'bin/tk-nn-train.py',
        'bin/tk-nn-infer.py'
    ],
    test_suite="tests",
    setup_requires=setup_requirements,
    tests_require=test_requirements,
    install_requires=INSTALL_REQUIRES,
    packages=find_packages(),
    zip_safe=False
)
