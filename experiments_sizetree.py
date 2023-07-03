import matplotlib.pyplot as plt
import matplotlib
import os
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.ensemble import IsolationForest
from size_tree.sizeRegressor_sorted import *
from sklearn.tree import DecisionTreeRegressor

if __name__ == '__main__':
    print('Using incremental algorithm!!!')
    dataset = 'kdd_other'
    dataset_path = os.path.join('./datasets/', dataset)
    df = pd.read_csv(dataset_path, header=0)
    y = df["label"]
    sample_size = y.value_counts()[1]
    X = df.drop("label", axis=1)
    X.columns = [int(x) for x in X.columns]
    print('~~~~~~~~~~~~~~~~~~~~~~dataset: ', dataset)
    print('n=', len(y), ' d=', len(X.columns), ' anomaly=', sample_size)

    black_ap = np.zeros(10)
    size_ap = np.zeros(10)
    counttime = 0
    reg_ap = np.zeros(10)
    for i in range(10):
        black_box = IsolationForest()
        black_box.fit(X)
        if_preds = black_box.decision_function(X)
        anomalyscores = 0.5 - if_preds
        black_ap[i] = average_precision_score(y, anomalyscores)
        import time
        starttime = time.time()
        explainer = SizeBasedRegressionTree(max_nodes=64)
        explainer.fit(anomalyscores, X)
        counttime += time.time()-starttime
        explainer_predict = explainer.predict_train()
        size_ap[i] = average_precision_score(y, explainer_predict)

        explainer = DecisionTreeRegressor(max_depth=5)
        explainer.fit(X, anomalyscores)
        reg_scores = explainer.predict(X)
        reg_ap[i] = average_precision_score(y,reg_scores)

    print('IF ap: %.3f, Our ap: %.3f, CART ap: %.3f, time per run: %.3f s'%(black_ap.mean(), reg_ap.mean(), size_ap.mean(), counttime/10))









