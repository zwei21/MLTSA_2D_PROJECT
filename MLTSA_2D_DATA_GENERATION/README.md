# MLTSA 2D DATA GENERATION
A repo to store the data generation code implemented by Zhengxiang Wei
#### Interface: TwoD_pot_data.py
----
The interface contains all the functions that user could import and use directly for the MLTSA 2D Data generation, no need to touch the dirty code in src to enjoy all the functionality of built methods.

Using example:
```Python
from TwoD_pot_data import *
```

Including the following methods:
- Potential Generation
  - pot_generator
  - show_pot_attributes
  - get_pot_func
- Trajectory Generation
  - genearte_traj
- Data Preprocessing
  - data_processor
  - data_process_full
- Data Projection(feature creation)
  - data_projector
  - data_projection

Full docstrings built-in sorce code, in case of having anything don't know how to use, just type
```Python
help(some_method)
```
And the helpful docstring would offer you instructions.

#### test: The examples using all the methods implemented..
----
in the test folder, we implemented four jupyter notebooks to test and present the examples using 2D generation code, you can check and read through the jupyter notebooks to have a direct view of what have been done.

Just in case you are new here, please read the notebooks in order of
1. potential generation
2. trajectory generation
3. data preprocessing
4. data projection

To provide you with the most smooth discovery experience with MLTSA 2D dataset generation examples.

#### src: The dirty works..
----
in the src folder, source code of 2D data generation are stored and called by the interface script we introduced before, there is no need to call into the src if you only requires minimum use of MLTSA 2D data, but if you want to modify the dataset generation yourself, feel free to dive into the src, I am sure that the well commented code provides possibilities for anyone to understand and play with the code.


Enjoy the two-D dataset and play with it!!
