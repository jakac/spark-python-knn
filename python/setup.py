# Python setup file. An example can be found at:
# https://github.com/pypa/sampleproject/blob/master/setup.py

from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand
import sys

class PyTest(TestCommand):

    user_options = [('pytest-args=', 'a', "Arguments to pass to py.test")]

    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.pytest_args = ['-s']

    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        #import here, cause outside the eggs aren't loaded
        import pytest
        errno = pytest.main(self.pytest_args)
        sys.exit(errno)


setup(name='gaussalgo-spark-python-knn',
      version='0.0.1',
      description='Function for computing K-NN in Apache Spark',
      author='Matej Jakimov',
      author_email='jakimov@gaussalgo.com',
      url='https://github.com/jakac/spark-python-knn',
      packages=find_packages(),
      install_requires=[
          'numpy',
          'scikit-learn'
      ],
      tests_require=['pytest'],
      cmdclass={'test': PyTest},
      namespace_packages=['gaussalgo']
     )