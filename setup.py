from setuptools import Extension, setup

# Defining the C extension module
module = Extension("symnmf", sources=['symnmfmodule.c', 'symnmf.c'])

setup(name='symnmf',
      version='1.0',
      description='Python wrapper for symNMF C extension',
      ext_modules=[module])