## Setup all variables
PYTHON = python3
PDFLATEX = pdflatex
BIBTEX = bibtex
BASH = bash

eval:
	PYTHONPATH="/home/shino/Uni/master_thesis" TF_CPP_MIN_LOG_LEVEL=2 $(PYTHON) sources/build/eval.py 1 8 32 1 32 75 1 0

anomaly:
	PYTHONPATH="/home/shino/Uni/master_thesis" TF_CPP_MIN_LOG_LEVEL=2 $(PYTHON) sources/build/eval_anomaly.py 0 8 32 1 32 75 1 1

graphs:
	PYTHONPATH="/home/shino/Uni/master_thesis" TF_CPP_MIN_LOG_LEVEL=2 $(PYTHON) sources/build/eval_graphs.py 0 8 32 1 32 75 1 1

reeval:
	PYTHONPATH="/home/shino/Uni/master_thesis" TF_CPP_MIN_LOG_LEVEL=2 $(PYTHON) sources/build/eval.py 0 8 16 1 16 75 1 0
	PYTHONPATH="/home/shino/Uni/master_thesis" TF_CPP_MIN_LOG_LEVEL=2 $(PYTHON) sources/build/eval_anomaly.py 0 8 16 1 16 75 1 0
	PYTHONPATH="/home/shino/Uni/master_thesis" TF_CPP_MIN_LOG_LEVEL=2 $(PYTHON) sources/build/eval_graphs.py 0 8 16 1 16 75 1 0

eval_all:
	nice -19 $(BASH) eval_configurations.sh

demo:
	PYTHONPATH="/home/shino/Uni/master_thesis" MPLBACKEND="webagg" TF_CPP_MIN_LOG_LEVEL=2 $(PYTHON) sources/build/demo.py 1 eval_0_DT_8_16_KNN_1_16_75_DS_1 /home/shino/Uni/master_thesis/bin/data/simple_square_anomaly.csv 0