import os
from setuptools import setup, find_packages

NAME = "recruitment_agency_detector"
VERSION = os.environ.get("RECRUITMENT_AGENCY_DETECTOR", '0.0.0')


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
    test_suite="tests",
    setup_requires=setup_requirements,
    tests_require=test_requirements,
    packages=find_packages(),
    zip_safe=False
)
