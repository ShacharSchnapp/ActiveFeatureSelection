This archive contains all the code that produces our experiments as presented in our paper:
S. Schnapp, S. Sabato, "Active Feature Selection for the Mutual Information Criterion", AAAI 2021, to appear.
To run the experiments for actively estimating the mutual information of a single feature, follow the instructions in the README.txt file in the "active_estimation_for_a_single_feature" folder.
To run the experiments for active feature selection, follow the instructions in the README.txt file in the "active _feature_selection_algorithm" folder.

-----------------
Environment
-----------------

Before you run the code, check that you have python 3.6 installed on you device.
Then go to code directory and set the PYTHONPATH to the code directory using the following command line:
sudo export PYTHONPATH="./" 

----------------------
Module dependencies
----------------------

python3.6 -m pip install scipy
python3.6 -m pip install numpy
python3.6 -m pip install matplotlib
python3.6 -m pip install sklearn
python3.6 -m pip install torch
