from setuptools import setup, find_packages
import sys
from setuptools.command.test import test as TestCommand


class PyTest(TestCommand):
    user_options = [('pytest-args=', 'a', "Arguments to pass to pytest")]

    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.pytest_args = ""

    def run_tests(self):
        # import here, cause outside the eggs aren't loaded
        import shlex
        import pytest
        self.pytest_args += " --cov=EMaligner --cov-report html "\
                            "--junitxml=test-reports/test.xml"

        errno = pytest.main(shlex.split(self.pytest_args))
        sys.exit(errno)


with open('test_requirements.txt', 'r') as f:
    test_required = f.read().splitlines()


with open('requirements.txt', 'r') as f:
    required = f.read().splitlines()


setup(name="nway_recover",
      description=("building up a repo from the ophys_ophys_nway "
                   "running in LIMS, Aug 2019"),
      author="Daniel Kapner, original code Fuhui Long",
      author_email='danielk@alleninstitute.org',
      url="https://github.com/AllenInstitute/nway_recover",
      packages=find_packages(),
      setup_requires=['setuptools_scm'],
      install_requires=required,
      tests_require=test_required,
      cmdclass={'test': PyTest})
