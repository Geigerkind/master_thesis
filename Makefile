## Setup all variables
PYTHON = python3
PDFLATEX = pdflatex
BIBTEX = bibtex
BASH = bash

eval:
	PYTHONPATH="/home/shino/Uni/master_thesis" TF_CPP_MIN_LOG_LEVEL=2 $(PYTHON) sources/build/eval.py 0 8 16 1 16 75 1 0

anomaly:
	PYTHONPATH="/home/shino/Uni/master_thesis" TF_CPP_MIN_LOG_LEVEL=2 $(PYTHON) sources/build/eval_anomaly.py 0 8 16 1 16 75 1 0

graphs:
	PYTHONPATH="/home/shino/Uni/master_thesis" TF_CPP_MIN_LOG_LEVEL=2 $(PYTHON) sources/build/eval_graphs.py 0 8 16 1 16 75 1 0

reeval:
	PYTHONPATH="/home/shino/Uni/master_thesis" TF_CPP_MIN_LOG_LEVEL=2 $(PYTHON) sources/build/eval.py 0 8 16 1 16 75 1 0
	PYTHONPATH="/home/shino/Uni/master_thesis" TF_CPP_MIN_LOG_LEVEL=2 $(PYTHON) sources/build/eval_anomaly.py 0 8 16 1 16 75 1 0
	PYTHONPATH="/home/shino/Uni/master_thesis" TF_CPP_MIN_LOG_LEVEL=2 $(PYTHON) sources/build/eval_graphs.py 0 8 16 1 16 75 1 0

eval_all:
	nice -19 $(BASH) eval_configurations.sh