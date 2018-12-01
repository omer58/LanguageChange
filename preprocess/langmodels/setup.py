from distutils.core import setup
from Cython.Build import cythonize

setup(
	name="cython_app",
	ext_modules = cythonize("train_word2vec.py")
)
