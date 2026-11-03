set -e
export EW_OUT=${EW_OUT:-./out}; mkdir -p $EW_OUT/fig
python experiments.py main errmodels sigmahat pareto ablation heterogeneity scale solvers fcMA fcXGB fcRF fcLSTM fcmerge
python analysis.py
