import matplotlib.pyplot as plt
import matplotlib
import os
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from rhf import RHF
import argparse
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import IsolationForest
from size_tree.sizeRegressor_sorted_accurate import *

def run_regressor_ranking(config):
    dataset = config.dataset
    num_iterations = config.num_iterations
    rhf_num_trees = config.rhf_num_trees
    method = config.methods

    dataset_path = os.path.join('./datasets/klf/', dataset)
    # dataset_path = os.path.join('./datasets/imbalance_AD/', dataset)
    df = pd.read_csv(dataset_path, header=0)
    # df = df.sample(frac=1).reset_index(drop=True)
    y = df["label"]
    # print(y.value_counts())
    sample_size = y.value_counts()[1]
    X = df.drop("label", axis=1)
    X.columns = [int(x) for x in X.columns]
    print('~~~~~~~~~~~~~~~~~~~~~~dataset: ', dataset)
    print('n=', len(y), ' d=', len(X.columns), ' anomaly=', sample_size)


    best_ap = {}
    best_ap[config.black+'_ap'] = []
    best_ap['%s_leaves'%config.black] = []

    if config.black == 'if':
        best_ap['reg_ap'] = []
        best_ap['reg_depth'] = []
        best_ap['reg_leaves'] = []
        best_ap['size_ap'] = []
        best_ap['size_leaves'] = []
        best_ap['size_depth'] = []
        best_ap['size_nodes'] = []
        best_ap['size_time'] = []
    elif config.black == 'rhf':
        best_ap[method + '_ap'] = []
        best_ap[method + '_leaves'] = []
        best_ap[method + '_depth'] = []
        if method == 'size':
            best_ap['size_nodes'] = []
            best_ap['size_time'] = []

    seed_state_list = [10010,20020,300, 90000,1,2,3,4,5,6]
    time_count = []
    for d in range(num_iterations):
        if config.nth != -1:
            d = config.nth
        ### RHF in CYTHON VERSION
        if config.black == 'rhf':
            black_box = RHF(num_trees=rhf_num_trees, max_height=5, seed_state=seed_state_list[d], check_duplicates=True, decremental=False,
                            use_kurtosis=True)
            black_box.fit(X)
            rhf_scores, scores_all = black_box.get_scores()
            black_leaves = np.sum([np.unique(x).size for x in scores_all])
            anomalyscores = rhf_scores / rhf_num_trees

        elif config.black == 'if':
            black_box = IsolationForest(random_state=seed_state_list[d])
            black_box.fit(X)
            anomalyscores = 0.5 - black_box.decision_function(X)
            black_leaves = np.sum([x.get_n_leaves() for x in black_box.estimators_])

        ### draw 3d figure
        # fig = plt.figure()
        # ax = fig.add_subplot()#projection='3d')
        # ax.scatter(X[y==0][0], anomalyscores[y==0], marker='+', c='b')
        # ax.scatter(X[y==1][0], anomalyscores[y==1], marker='+', c='r')
        # ax.set_xlabel('y')
        # ax.set_ylabel('rhf_score')
        # # ax.set_zlabel('rhf_score')
        # plt.savefig('./results/sizebased_method_sorted/img/2d_kdd_http_distinct_rhf_f0.png')
        # plt.cla()
        # plt.hist(anomalyscores[y==0], color='b', bins=100, density=False)
        # plt.hist(anomalyscores[y==1], color='r', bins=100, density=False)
        # plt.ylim([0,50])
        # plt.savefig('./results/sizebased_method_sorted/img/kdd_http_distinct_gt_dis.png')
        # plt.cla()
        # exit()

        black_ap = average_precision_score(y, anomalyscores)
        best_ap['%s_ap'%config.black].append(black_ap)
        best_ap['%s_leaves'%config.black].append(black_leaves)

        # EXPLAINER Cart Tree
        if config.black =='if' or method == 'reg':
            import time
            start = time.time()
            explainer = DecisionTreeRegressor(max_depth=3)
            explainer.fit(X, anomalyscores)
            time_count.append(time.time() - start)
            reg_scores = explainer.predict(X)
            reg_ap = average_precision_score(y, reg_scores)
            reg_depth = explainer.get_depth()
            reg_leaves = explainer.get_n_leaves()
            best_ap['reg_ap'].append(reg_ap)
            best_ap['reg_depth'].append(reg_depth)
            best_ap['reg_leaves'].append(reg_leaves)
            if config.vis == True:
                from sklearn import tree
                import collections
                import pydotplus
                import graphviz
                feature_name = ['x%d'%x for x in X.columns]
                if dataset in ['kdd_ftp', 'kdd_finger']:
                    feature_name = ['duration', 'source bytes', 'destination bytes']
                elif dataset == 'wbc':
                    feature_name = ['radius'
                                    ,'texture'
                                    ,'perimeter'
                                    ,'area'
                                    ,'smoothness'
                                    ,'compactness'
                                    ,'concavity'
                                    ,'concave points'
                                    ,'symmetry'
                                    ,'fractal dimension']*3
                dot_data = tree.export_graphviz(explainer, feature_names=feature_name,
                                                           impurity=False,
                                                           out_file=None,
                                                           node_ids=True,
                                                           filled=True,
                                                           rounded=True)
                graph = pydotplus.graph_from_dot_data(dot_data)
                colors = ('turquoise', 'orange')
                edges = collections.defaultdict(list)
                for edge in graph.get_edge_list():
                    edges[edge.get_source()].append(int(edge.get_destination()))
                internal_node_index = [int(x) for x in edges.keys()]
                rootnode = graph.get_node('0')[0]
                rootnode.set_fontsize(40)
                leafinfo = rootnode.get_label().strip('"')
                leafinfo = leafinfo.split('\\')[:-1]
                leafinfo = '\\'.join(leafinfo)
                new_info = '"%s"' % (leafinfo)
                rootnode.set_label(new_info)
                print(new_info)
                for edge in edges:
                    edges[edge].sort()

                    for i in range(2):
                        dest = graph.get_node(str(edges[edge][i]))[0]
                        dest.set_fontsize(40)

                        if edges[edge][i] in internal_node_index:
                            destinfo = dest.get_label().strip('"')
                            destinfo = destinfo.split('\\')[:-1]
                            destinfo = '\\'.join(destinfo)
                            new_info = '"%s"' % (destinfo)
                            dest.set_label(new_info)
                            dest.set_fillcolor("#ffffff")
                        else:
                            # leaf node
                            leafinfo = dest.get_label().strip('"')
                            new_info = leafinfo.replace('value', 'anomaly_score')
                            # node_index = int(leafinfo.split('\\')[0].split('#')[-1])
                            # n_sample = int(leafinfo.split('\\')[1].split(' ')[-1])
                            # import ipdb;ipdb.set_trace()
                            # data_index = explainer.leaf_gt[np.where(np.array(explainer.leaf_nodes) == node_index)[0][0]]
                            # if len(data_index) != n_sample:
                            #     print('Error! the samples are not equal to data index!')
                            #     import ipdb;ipdb.set_trace()
                            # leaf_gt = y[data_index]
                            # # print(leaf_gt.value_counts())
                            # anomaly_rate = leaf_gt.sum() / n_sample
                            # new_info = '"%s\\%s"' % (leafinfo, 'nanomaly_ratio = %.2f'%anomaly_rate)

                            dest.set_label(new_info)
                            # dest.set_fillcolor(colors[1])

                graph.write_pdf('./imgtest/sizetree/%s_%stree.pdf'%(dataset, method))
                print('Img saved in %s!' % './imgtest/sizetree/%s_%stree.pdf'%(dataset, method))

        # EXPLAINER:
        if method =='size':
            import time
            start = time.time()
            if args.inc:
                explainer = SizeBasedRegressionTree(max_depth=20, max_nodes=64, score_way='RHF', epslon=1.)
            else:
                explainer = SizeBasedRegressionTree(max_depth=20, score_way='RHF')#, epslon=config.epslon)
            explainer.fit(anomalyscores, X)
            time_count.append(time.time() - start)
            explainer_predict = explainer.predict_train()

            size_ap = average_precision_score(y, explainer_predict)
            best_ap['size_ap'].append(size_ap)
            best_ap['size_depth'].append(explainer.layer + 1)
            best_ap['size_leaves'].append(len(explainer.leaf_nodes))
            best_ap['size_nodes'].append(len(explainer.all_nodes))
            best_ap['size_time'].append(time.time() - start)

            if config.vis == True:
                explainer.get_attribute()

                from sklearn import tree
                import collections
                import pydotplus
                import graphviz
                feature_name = ['x%d'%x for x in X.columns]
                if dataset in ['kdd_ftp', 'kdd_finger']:
                    feature_name = ['duration', 'source bytes', 'destination bytes']
                elif dataset == 'breastcancer':
                    feature_name = ['Clump Thickness',
                                    'Uniformity of Cell Size',
                                    'Uniformity of Cell Shape',
                                    'Marginal Adhesion',
                                    'Single Epithelial Cell Size',
                                    'Bare Nuclei',
                                    'Bland Chromatin',
                                    'Normal Nucleoli',
                                    'Mitoses']
                elif dataset == 'wbc':
                    feature_name = ['radius'
                                    ,'texture'
                                    ,'perimeter'
                                    ,'area'
                                    ,'smoothness'
                                    ,'compactness'
                                    ,'concavity'
                                    ,'concave points'
                                    ,'symmetry'
                                    ,'fractal dimension']*3
                dot_data = tree.export_graphviz(explainer, feature_names=feature_name,
                                                           impurity=False,
                                                           out_file=None,
                                                           node_ids=True,
                                                           filled=True,
                                                           rounded=True)
                graph = pydotplus.graph_from_dot_data(dot_data)
                colors = ('turquoise', 'orange')
                edges = collections.defaultdict(list)
                for edge in graph.get_edge_list():
                    edges[edge.get_source()].append(int(edge.get_destination()))
                internal_node_index = [int(x) for x in edges.keys()]
                rootnode = graph.get_node('0')[0]
                rootnode.set_fontsize(50)
                leafinfo = rootnode.get_label().strip('"')
                leafinfo = leafinfo.replace('\\nvalue = -1.0', '')
                # print('after: ', leafinfo)
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
                            # leaf_score = float(leafinfo.split('\\')[2].split(' ')[-1])
                            data_index = explainer.leaf_gt[np.where(np.array(explainer.leaf_nodes) == node_index)[0][0]]
                            if len(data_index) != n_sample:
                                print('Error! the samples are not equal to data index!')
                                import ipdb;ipdb.set_trace()
                            leaf_gt = y[data_index]
                            # print('---before: ', leafinfo)
                            anomaly_rate = leaf_gt.sum() / n_sample
                            leafinfo = '\\'.join(leafinfo.split('\\')[:-1])
                            # leafinfo = leafinfo.replace('value = %.3f'%leaf_score, 'nanomaly_score = %.3f'%(leaf_score/100))
                            # print('after: ', leafinfo)
                            # import ipdb;ipdb.set_trace()
                            newinfo = '"%s\\%s\\%s"' % (leafinfo, 'nanomaly_score = %.3f'%(1/n_sample), 'nanomaly_ratio = %.2f'%anomaly_rate)
                            # print('new: ', newinfo)
                            dest.set_label(newinfo)
                            # dest.set_fillcolor(colors[1])
                imgpath = './imgtest/sizetree/%s_%stree_scoresize.pdf'%(dataset, method)
                graph.write_pdf(imgpath)
                imgpath = './imgtest/sizetree/%s_%stree.png'%(dataset, method)
                graph.write_png(imgpath)
                print('Img saved in %s!' % imgpath)

        if config.nth != -1:
            break

    # print('%s AP = : %.4f ' % (config.black, np.mean(best_ap[config.black + '_ap'])))
    for me in best_ap.keys():
        print(me, ': %.4f' % np.mean(best_ap[me]))
    # print(best_ap)
    data = pd.DataFrame(best_ap)
    if config.nth != -1:
        data['ID'] = config.nth
        data['Datasets'] = [dataset]
    else:
        data['ID'] = np.arange(num_iterations)
        data['Datasets'] = [dataset] * num_iterations
    print('%s takes %.4f s for each tree on %s!' % (method, np.mean(time_count), dataset))
    return data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='kdd_other', type=str)
    args = parser.parse_args()

    print(' Using incremental algorithm!!!')

    dataset = args.dataset
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
        #explainer = SizeBasedRegressionTree(max_nodes=64)
        explainer = SizeBasedRegressionTree(max_depth=20, max_nodes=64, score_way='RHF', epslon=1.)
        explainer.fit(anomalyscores, X)
        counttime += time.time()-starttime
        explainer_predict = explainer.predict_train()
        size_ap[i] = average_precision_score(y, explainer_predict)

        explainer = DecisionTreeRegressor(max_depth=5)
        explainer.fit(X, anomalyscores)
        reg_scores = explainer.predict(X)
        reg_ap[i] = average_precision_score(y,reg_scores)

    print('IF ap: %.3f, Our ap: %.3f, CART ap: %.3f, time per run: %.3f s'%(black_ap.mean(),  size_ap.mean(), reg_ap.mean(),counttime/10))











