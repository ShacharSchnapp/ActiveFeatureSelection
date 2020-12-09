This folder provides the implementation of the Active Feature Selection algorithm described in the paper, and all the other algorithms that were compared to in the paper. It can be used to reproduce the experiment results reported in the paper.

For environment setup, see the README.txt in the root of the archive.


To run an baseline experiment which compares AFC, RANDOM, CORESET and DWUS:
python3.6 baseline.py [-h [HELP]]
                   [-path [PATH]]
                   [-label [TOTAL_LABELS]]
                   [-k [Features]]
                   [-delta [DELTA]]
		   [-Lambda [LAMBDA]]
                   [-n_threads [NUMBER_OF_THREADS]]
                   [-t_times [TIMES_PEER_THREAD]]

Example:  python3.6 baseline.py -path dataset.mat -k 5 -label 500 -delta 0.05 -Lambda 30 -n_threads 10 -t_times 3



To run an ablation test which compares all the different components of AFC algorithm (including AFS, RANDOM and the algorithms in the ablation studies), use the following command line:
python3.6 experiment.py [-h [HELP]]
                   [-path [PATH]]
                   [-label [TOTAL_LABELS]]
                   [-k [Features]]
                   [-delta [DELTA]]
		   [-Lambda [LAMBDA]]
                   [-n_threads [NUMBER_OF_THREADS]]
                   [-t_times [TIMES_PEER_THREAD]]

Example:  python3.6 ablation.py -path dataset.mat -k 5 -label 500 -delta 0.05 -Lambda 30 -n_threads 10 -t_times 3


To run an experiment which compares the 3 aggregation functions for the socre, use the following command line:
python3.6 aggregation.py [-h [HELP]]
                   [-path [PATH]]
                   [-label [TOTAL_LABELS]]
                   [-k [Features]]
                   [-delta [DELTA]]
		   [-Lambda [LAMBDA]]
                   [-n_threads [NUMBER_OF_THREADS]]
                   [-t_times [TIMES_PEER_THREAD]]
 
Example:  python3.6 aggregation.py -path dataset.mat -k 5 -label 500 -delta 0.05 -Lambda 30 -n_threads 10 -t_times 3 


To run the stand-alone AFS algorithm, use the following command line:
python3.6 run_afs.py [-h [HELP]]
	             [-path [PATH]]
		     [-label [TOTAL_LABELS]]
                     [-k [Features]]
                     [-delta [DELTA]]
		     [-Lambda [LAMBDA]]

Example:  python3.6 run_afs.py -path dataset.mat -k 5 -label 500 -delta 0.05 -Lambda 30


-------------------
optional arguments:
-------------------
  -h, --help     show this help message and exit

  -path [PATH]   The full path of the dataset file. This is a .mat file with variables X (Samples) and binary label Y (Labels)

  -label[LABEL]  The budget for labels (default: 120)

  -k [NUMBER]  The number of features to select (default: 5)

  -delta [DELTA]  The delta parameter of the algorithm (default: 0.05)

  -Lambda [LAMBDA]  The number of iterations for testing change. Set to inf to disable the safeguard. 

  -n_threads [NUMBER_OF_THREADS]  Number of experiment threads to run.

  -t_times [TIMES_PER_THREAD]  The number of times that each thread will repeat the test (default:3)


-----
Data:
-----
The dataset is a .mat file with variables X (Samples) and binary label Y (Labels). The X is a numpy.array of all the samples on the dataset, where each sample is a numpy.array of features. Feature values must be integers. No quantization or binning is done by the implementation. The Y is a numpy.array of zeros and ones.


-----------------------
Output of experiment.py
-----------------------
The overall number of tests that will run on the data set is NUMBER_OF_THREADS x TIMES_PER_THREAD. Each test runs each of the algorithms (SINGLE, AVG-ALL, AVG-SEL, AFS, RANDOM) once on the data set.
Two files will be generated in the folder path: ./results/dataset_name.
1. A png file which plots the average of the results
2. A pth file with the results of all the experiments. This file contains a dictionary, where the keys are the algorithm names and the values are a list of length  NUMBER_OF_THREADS x TIMES_PER_THREAD of vectors of length LABEL. Each coordinate in each vector gives the average mutual information with the label of the k  features that this algorithm selected in this run after this number of labels. Note that the reported mutual information is calculated based on the entire data set.

-----------------------
Output of score.py
-----------------------
The output format is the same as that of experiment.py. The only difference is that each test runs the algorithms AFS (With l1 aggregation), AFS with l2 aggregatin, AFS with l-infinity aggregation.

-----------------------
Output of AFS algorithm
-----------------------
Prints the k features selected by the AFS algorithm. The features are identified by their order in the feature vector, where the first feature is numbered 0.



