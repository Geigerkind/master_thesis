## Setup all variables
PYTHON = python3
PDFLATEX = pdflatex
BIBTEX = bibtex
BASH = bash

eval:
	PYTHONPATH="/home/shino/Uni/master_thesis" $(PYTHON) sources/build/eval.py 0 16 20 1 30 75

graphs:
	PYTHONPATH="/home/shino/Uni/master_thesis" $(PYTHON) sources/build/eval_graphs.py eval_0_DT_16_20_KNN_1_30_75

reeval:
	PYTHONPATH="/home/shino/Uni/master_thesis" $(PYTHON) sources/build/eval.py 0 16 20 1 30 75
	PYTHONPATH="/home/shino/Uni/master_thesis" $(PYTHON) sources/build/eval_graphs.py eval_0_DT_16_20_KNN_1_30_75