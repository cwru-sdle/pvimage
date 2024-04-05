import os

# PyPI
# This is for the real PyPI - the Python Package Index
# Be sure to have PyPI setup completed (e.g. ~/.pypirc file):
#   https://packaging.python.org/distributing/#uploading-your-project-to-pypi

# Initial registration command (run only once)
#os.system("python setup.py register")
# OR (more secure)
#os.system("twine register dist/*.whl"

# Twine upload of files (after registration)
#os.system("twine upload dist/*")


# PyPI Test Server
# This is for the Testing PyPI - the Testing Python Package Index
# Be sure to have PyPI setup completed (e.g. ~/.pypirc file):
#   https://wiki.python.org/moin/TestPyPI

# Initial registration command (run only once)
#os.system("python setup.py register -r https://testpypi.python.org/pypi")

# Twine upload of files (after registration)
os.system("twine upload dist/* -r testpypi")
