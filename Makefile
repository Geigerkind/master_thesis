## Setup all variables
PYTHON = python3
PDFLATEX = pdflatex
BIBTEX = bibtex
BASH = bash

eval:
	PYTHONPATH="/home/shino/Uni/master_thesis" TF_CPP_MIN_LOG_LEVEL=2 $(PYTHON) sources/build/eval.py 1 8 16 1 16 75 1

anomaly:
	PYTHONPATH="/home/shino/Uni/master_thesis" TF_CPP_MIN_LOG_LEVEL=2 $(PYTHON) sources/build/eval_anomaly.py eval_1_DT_8_16_KNN_1_16_75_DS_1

graphs:
	PYTHONPATH="/home/shino/Uni/master_thesis" TF_CPP_MIN_LOG_LEVEL=2 $(PYTHON) sources/build/eval_graphs.py eval_0_DT_16_20_KNN_1_30_75_DS_1

reeval:
	PYTHONPATH="/home/shino/Uni/master_thesis" TF_CPP_MIN_LOG_LEVEL=2 $(PYTHON) sources/build/eval.py 0 16 20 1 30 75 1
	PYTHONPATH="/home/shino/Uni/master_thesis" TF_CPP_MIN_LOG_LEVEL=2 $(PYTHON) sources/build/eval_anomaly.py eval_0_DT_16_20_KNN_1_30_75_DS_1
	PYTHONPATH="/home/shino/Uni/master_thesis" TF_CPP_MIN_LOG_LEVEL=2 $(PYTHON) sources/build/eval_graphs.py eval_0_DT_16_20_KNN_1_30_75_DS_1

eval_all:
	nice -19 $(BASH) eval_configurations.sh