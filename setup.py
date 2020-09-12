from setuptools import setup, find_packages


with open('requirements.txt', 'r') as f:
    required = f.read().splitlines()


setup(name="ophys_nway_matching",
      use_scm_version=True,
      description=("building up a repo from the ophys_ophys_nway "
                   "running in LIMS, Aug 2019"),
      author="Daniel Kapner, original code Fuhui Long",
      author_email='danielk@alleninstitute.org',
      url="https://github.com/AllenInstitute/ophys_nway_matching",
      packages=find_packages(),
      setup_requires=['setuptools_scm'],
      install_requires=required)
