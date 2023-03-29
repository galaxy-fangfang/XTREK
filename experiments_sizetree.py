import pickle as pkl
import matplotlib.pyplot as plt
import matplotlib
import os
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.ensemble import IsolationForest
from size_tree.sizeRegressor_sorted import *

def vistree(explainer, dataset):
    explainer.get_attribute()

    from sklearn import tree
    import collections
    import pydotplus
    feature_name = ['x%d'%x for x in X.columns]
    if dataset == 'breastcancer':
        feature_name = ['Clump Thickness',
                        'Uniformity of Cell Size',
                        'Uniformity of Cell Shape',
                        'Marginal Adhesion',
                        'Single Epithelial Cell Size',
                        'Bare Nuclei',
                        'Bland Chromatin',
                        'Normal Nucleoli',
                        'Mitoses']

    dot_data = tree.export_graphviz(explainer, feature_names=feature_name,
                                                impurity=False,
                                                out_file=None,
                                                node_ids=True,
                                                filled=True,
                                                rounded=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
    edges = collections.defaultdict(list)
    for edge in graph.get_edge_list():
        edges[edge.get_source()].append(int(edge.get_destination()))
    internal_node_index = [int(x) for x in edges.keys()]
    rootnode = graph.get_node('0')[0]
    rootnode.set_fontsize(50)
    leafinfo = rootnode.get_label().strip('"')
    leafinfo = leafinfo.replace('\\nvalue = -1.0', '')
    new_info = '"%s"' % (leafinfo)
    rootnode.set_label(new_info)
    for edge in edges:
        edges[edge].sort()
        
        for i in range(2):
            dest = graph.get_node(str(edges[edge][i]))[0]
            dest.set_fontsize(50)
            if edges[edge][i] in internal_node_index:
                leafinfo = dest.get_label().strip('"')
                leafinfo = leafinfo.replace('\\nvalue = -1.0', '')
                new_info = '"%s"' % (leafinfo)
                dest.set_label(new_info)
                dest.set_fillcolor("#ffffff")
            else:
                # leaf node
                leafinfo = dest.get_label().strip('"')
                node_index = int(leafinfo.split('\\')[0].split('#')[-1])
                n_sample = int(leafinfo.split('\\')[1].split(' ')[-1])
                data_index = explainer.leaf_gt[np.where(np.array(explainer.leaf_nodes) == node_index)[0][0]]
                if len(data_index) != n_sample:
                    print('Error! the samples are not equal to data index!')
                    exit(0)
                leaf_gt = y[data_index]
                anomaly_rate = leaf_gt.sum() / n_sample
                leafinfo = '\\'.join(leafinfo.split('\\')[:-1])
                newinfo = '"%s\\%s\\%s"' % (leafinfo, 'nanomaly_score = %.3f'%(1/n_sample), 'nanomaly_ratio = %.2f'%anomaly_rate)
                dest.set_label(newinfo)
    imgpath = '%s.png'%dataset
    graph.write_png(imgpath)
    print('Img saved in %s!' % imgpath)


if __name__ == '__main__':
    print(' Using incremental algorithm!!!')
    dataset = 'breastcancer'
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
    for i in range(10):
        black_box = IsolationForest()
        black_box.fit(X)
        anomalyscores = 0.5 - black_box.decision_function(X)
        black_ap[i] = average_precision_score(y, anomalyscores)

        explainer = SizeBasedRegressionTree(max_nodes=64)
        explainer.fit(anomalyscores, X)
        explainer_predict = explainer.predict_train()
        size_ap[i] = average_precision_score(y, explainer_predict)
        if i == 0:
            vistree(explainer, dataset)
    print('IF ap: %.3f, Our ap: %.3f'%(black_ap.mean(), size_ap.mean()))
    









