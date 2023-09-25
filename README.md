# XTREK
This repository contains the source code of XTreK and the datasets used in the experiments presented in the paper "Tree-based Kendall’s τ Maximization for Explainable Unsupervised Anomaly Detection". This paper has been accepted at the 23rd IEEE International Conference on Data Mining (ICDM 2023).


**install**
1. Install package: `scikit-learn v. 1.1.2`, `scipy v.1.6.2`, `sortedcontainers v.2.4.0`, and `numpy v.1.21.6`
2. If you want to run the unsupervised anomaly detection algorithm: Random Histogram Forest, please refer to the official repo to install: <https://github.com/anrputina/rhf>
3. Run experiments: `python experiments_accurate.py --dataset kdd_other`, dataset `kdd_other` is in the folder: `datasets/`

Usage:
    from size_tree.XTREK import *
    explainer = SizeBasedRegressionTree(max_depth=20, max_nodes=64)
    explainer.fit(anomalyscores, X)
    explainer_predict = explainer.predict_train()

