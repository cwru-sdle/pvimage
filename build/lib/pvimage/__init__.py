"""
pvimage package is for building a pipeline that can read raw images,
raw images can be processed and then fed to machine learning
algorithms for classification. The two major approaches are supervised
machine learning and unsupervised machine learning approaches.
"""

import pvimage.process, pvimage.pipelines

def readme():
  """This function displays the contents of the README.rst file.

  Args:
      NULL (NA): There are no parameters.

  Returns:
    NULL: There are no returns, a print statement is executed.
  """
  import os
  this_dir, this_filename = os.path.split(__file__)
  DATA_PATH = os.path.join(this_dir, "../README.rst")
  with open(DATA_PATH) as f:
      print(f.read())



