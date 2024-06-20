import sys
from train import *
from fluency_model import *
from test import *
from fluency_test import *

# main
if __name__ == "__main__":
    file1 = sys.argv[2]
    file2 = sys.argv[3]

    if sys.argv[1] == "train":
        trainer_fluency(file1, file2)
        trainer_function(file1, file2)

    elif sys.argv[1] == "test":
        tester_fluency(file1, file2)
        tester_function(file1, file2)
