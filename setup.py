from setuptools import setup

import os
BASEPATH = os.path.dirname(os.path.abspath(__file__))

setup(name='ganrl',
      py_modules=['ganrl'],
      install_requires=[
          'tensorflow'
      ],
)
