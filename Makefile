## Setup all variables
PYTHON = python3
PDFLATEX = pdflatex
BIBTEX = bibtex
BASH = bash

eval:
	PYTHONPATH="/home/shino/Uni/master_thesis" $(PYTHON) sources/build/eval.py

graphs:
	PYTHONPATH="/home/shino/Uni/master_thesis" $(PYTHON) sources/build/eval_graphs.py

reeval:
	PYTHONPATH="/home/shino/Uni/master_thesis" $(PYTHON) sources/build/eval.py
	PYTHONPATH="/home/shino/Uni/master_thesis" $(PYTHON) sources/build/eval_graphs.py