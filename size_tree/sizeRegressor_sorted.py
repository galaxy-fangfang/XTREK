"""Main module."""
import collections
import os
import numpy as np
import pandas as pd
import scipy.stats as stats
import math
from math import log

def H_correct(n):
    return sum(1/d for d in range(1,n+1))

Harmonic_n = np.array([H_correct(n) for n in range(1,101+1)])

def H(n):
    # Euler-Mascheroni constant
    gamma = 0.57721566490153286060651209008240243104215933593992
    if n > 500:
        return gamma + log(n)
    elif n > 100:
        return gamma + log(n) + 0.5/n
    else:
        return Harmonic_n[n-1]

def H_func(w, n, N):
    y = H(N - n) + (N-n-2)*w[n]
    return y

def weighted_kendall(anomaly_scores, score_per_tree):
    tau, p_value = stats.weightedtau(anomaly_scores, score_per_tree, rank=None)
    return tau

def weighted_kendall_my(x, y, pointer, rank=None, weigher=None, tot=None, u=None, x_sorted=True, y_sorted=True):
    n = len(x)

    exchanges_weight = np.zeros(1, dtype=np.float64)
    # initial sort on values of x and, if tied, on values of y
    if x_sorted:
        perm = np.arange(n)
    else:
        perm = np.lexsort((y,x))
    temp = np.empty(n, dtype=np.intp) # support structure

    if weigher is None:
        weigher = lambda x: 1./(1 + x)

    def _invert_in_place(perm):
        for n in range(len(perm)-1, -1, -1):
            i = perm[n]
            if i < 0:
                perm[n] = -i - 1
            else:
                if i != n:
                    k = n
                    while True:
                        j = perm[i]
                        perm[i] = -k - 1
                        if j == n:
                            perm[n] = i
                            break

                        k = i
                        i = j

    if rank is None:
        # To generate a rank array, we must first reverse the permutation
        # (to get higher ranks first) and then invert it.
        rank = np.empty(n, dtype=np.intp)
        rank[...] = perm[::-1]
        _invert_in_place(rank)#np.argsort(np.argsort(-1*x))

    weights_x = weigher(rank[perm])

    # weigh joint ties
    first = 0
    t = 0
    w = weigher(rank[perm[first]])
    s = w

    for i in range(1, n):
        if x[perm[first]] != x[perm[i]] or y[perm[first]] != y[perm[i]]:
            t += s * (i - first - 1)
            first = i
            s = 0

        w = weigher(rank[perm[i]])
        s += w

    t += s * (n - first - 1)

    # weigh ties in x
    # u will be all the same for X
    if u is None:
        first = 0
        u = 0
        w = weigher(rank[perm[first]])
        s = w
        for i in range(1, n):
            if x[perm[first]] != x[perm[i]]:
                u += s * (i - first - 1)
                first = i
                s = 0

            w = weigher(rank[perm[i]])
            s += w

        u += s * (n - first - 1)
        if first == 0: # x is constant (all ties)
            return np.nan

    # this closure recursively sorts sections of perm[] by comparing
    # elements of y[perm[]] using temp[] as support

    def weigh(offset, length):
        if length == 1:
            return weigher(rank[perm[offset]])
        length0 = length // 2
        length1 = length - length0
        middle = offset + length0
        residual = weigh(offset, length0)
        weight = weigh(middle, length1) + residual
        if y[perm[middle - 1]] < y[perm[middle]]:
            return weight

        # merging
        i = j = k = 0 # j: left, k: right

        while j < length0 and k < length1:
            if y[perm[offset + j]] <= y[perm[middle + k]]:
                temp[i] = perm[offset + j]
                residual -= weigher(rank[temp[i]])
                j += 1
            else:
                temp[i] = perm[middle + k]
                exchanges_weight[0] += weigher(rank[temp[i]]) * (
                    length0 - j) + residual
                k += 1
            i += 1

        perm[offset+i:offset+i+length0-j] = perm[offset+j:offset+length0]
        perm[offset:offset+i] = temp[0:i]
        return weight

    def weigh_my(length, length0):
        if length == 1:
            return weigher(rank[perm[0]])
        length1 = length - length0
        residual = weigher(rank[perm[:length0]]).sum()
        # merging
        i = j = k = 0

        while j < length0 and k < length1:
            if y[perm[j]] <= y[perm[length0 + k]]:
                temp[i] = perm[j]
                residual -= weigher(rank[temp[i]])
                j += 1
            else:
                temp[i] = perm[length0 + k]
                exchanges_weight[0] += weigher(rank[temp[i]]) * (
                    length0 - j)  + residual
                k += 1
            i += 1

        perm[i:i+length0-j] = perm[j:length0]
        perm[0:i] = temp[0:i]

    # weigh discordances
    if y_sorted:
        weigh_my(n, pointer)
    else:
        weigh(0, n)

    # weigh ties in y
    first = 0
    v = 0
    w = weigher(rank[perm[first]])
    s = w

    for i in range(1, n):
        if y[perm[first]] != y[perm[i]]:
            v += s * (i - first - 1)
            first = i
            s = 0

        w = weigher(rank[perm[i]])
        s += w

    v += s * (n - first - 1)
    if first == 0: # y is constant (all ties)
        return np.nan

    # weigh all pairs
    if tot is None:
        s = 0
        for i in range(n):
            w = weigher(rank[perm[i]])
            s += w

        tot = s * (n - 1)

    tau = ((tot - (v + u - t)) - 2. * exchanges_weight[0]
           ) / np.sqrt(tot - u) / np.sqrt(tot - v)

    return min(1., max(-1., tau)), tot, u, v, t, exchanges_weight[0], weights_x

def binarySearch(a, low, high, item):
    # return the position for item to insert
    if (high <= low):
        # if exists same values, insert item behind so that the fewer weights will be changed
        return  low + 1 if item >= a[low] else low

    mid = (low + high) // 2

    if (item == a[mid]):
        return mid + 1

    if (item > a[mid]):
        return binarySearch(a, mid + 1, high, item)
    return binarySearch(a, low, mid - 1, item)

def get_best_split(y, X, low_bound, total_size, epslon=0.1):
    """
    Get attribute split according to Weight Kendall

    :param y: the label of the node
    :param X: the dataset of the node
    :returns:
        - feature_index: the attribute index to split
        - feature_split: the attribute value to split
    """

    # turn dataframe to array
    X = X.values
    y = y.values

    d = X.shape[1]
    n_samples = X.shape[0]
    best_feature = None
    best_value = None

    kendall_base = 0
    if len(np.unique(y)) == 1:
        return best_feature, best_value
    # Go through all features and all vaules
    y_inc = np.sort(y)

    # weights for kendall tau
    N_data = n_samples
    weights_bi = np.array([1/(N_data-n) for n in range(N_data)])
    sum_weights_bi = weights_bi.sum()

    for i in range(d):
        feature = i
        X_i_sorted = sorted(X[:, feature])
        X_i_sorted_unique = np.unique(sorted(X[:, feature]))
        # Feature values are all the same, no need to select splitting value
        if len(X_i_sorted_unique) == 1:
            continue

        X_i_sorted_index = np.lexsort((y, X[:, feature]))
        y_i_sorted_by_f = y[X_i_sorted_index]
        xx = np.lexsort((X[:, feature],y_i_sorted_by_f))
        xx_rank = np.argsort(xx)


        # Compute all the splitting values
        left_size = 0
        right_size = 0

        left_size_pre = 0
        left_size_pre_star = 0
        bFirst = True

        tot, u = None, None

        from sortedcontainers import SortedList
        left_y_sorted = SortedList()
        for j in range(len(X_i_sorted)):
            if j + 1 >= n_samples:
                break
            value = X_i_sorted[j] / 2.0 + X_i_sorted[j+1] / 2.0

            step = math.ceil(n_samples*epslon)
            if step==0 or j % step == 0:
                left_size_pre = 0
            if left_size_pre== 0:
                # sorted container
                left_y_sorted.add(y_i_sorted_by_f[j])
                if X_i_sorted[j] == X_i_sorted[j+1]:
                    continue
                left_size = j + 1
                right_size  = n_samples - left_size
                if left_size <= low_bound:
                    continue
                if right_size <= low_bound:
                    break

                # Getting the left and right ys
                right_y = y_i_sorted_by_f[left_size: ]
                from collections import deque
                right_y = deque(right_y)
                right_y_sorted = SortedList(right_y)

                """
                    aa == dd, bb == cc
                    aa = weighted_kendall(np.sort(y)[::-1], np.concatenate((np.sort(left_y)[::-1], np.sort(right_y)[::-1]), axis=None))
                    bb = weighted_kendall(np.sort(y)[::-1], np.concatenate((np.sort(right_y)[::-1], np.sort(left_y)[::-1]), axis=None))
                    cc = weighted_kendall(np.sort(y), np.concatenate((np.sort(left_y), np.sort(right_y)), axis=None))
                    dd = weighted_kendall(np.sort(y), np.concatenate((np.sort(right_y), np.sort(left_y)), axis=None))

                    k_dec = weighted_kendall(np.sort(y)[::-1], np.concatenate((np.sort(left_y)[::-1], np.sort(right_y)[::-1]), axis=None))
                        = weighted_kendall(np.sort(y), np.concatenate((np.sort(right_y), np.sort(left_y)), axis=None))
                    k_inc = weighted_kendall(np.sort(y), np.concatenate((np.sort(left_y), np.sort(right_y)), axis=None))
                """

                concat_dec = np.concatenate((right_y_sorted, left_y_sorted), axis=None)
                concat_inc = np.concatenate((left_y_sorted, right_y_sorted), axis=None)

                k_dec, tot, u, v_dec, t_dec, exchange_dec, weights_ori = weighted_kendall_my(y_inc, concat_dec, right_size, tot=tot, u=u)
                k_inc, tot, u, v_inc, t_inc, exchange_inc, weights_ori = weighted_kendall_my(y_inc, concat_inc, left_size, tot=tot, u=u)

                weights_ori_left = weights_ori[:left_size]
                weights_ori_right = weights_ori[left_size:]
                weights_ori_left_1 = weights_ori[:right_size]
                weights_ori_right_1 = weights_ori[right_size:]

            else:
                insert_item = right_y.popleft()
                insert_index = left_y_sorted.bisect_right(insert_item)
                original_index = right_y_sorted.index(insert_item)
                _ = right_y_sorted.pop(original_index)

                #exchange for left to right
                insert_item_weight_old_inc = weights_ori_right[original_index]
                insert_item_weight_new_inc = weights_ori_left[insert_index-1 if insert_index == j else insert_index]

                left_exchange_delta = (j - insert_index) * insert_item_weight_old_inc + weights_ori_left[insert_index : j].sum()
                right_exchange_delta = insert_item_weight_new_inc * (original_index) + weights_ori_right[: original_index].sum()

                exchange_inc = exchange_inc - left_exchange_delta  + right_exchange_delta

                #exchange for right to left
                insert_item_weight_old_dec = weights_ori_left_1[original_index]
                insert_item_weight_new_dec = weights_ori_right_1[insert_index-1 if insert_index == j else insert_index]

                left_exchange_delta = (n_samples - j - original_index) * insert_item_weight_new_dec + weights_ori_left_1[original_index : n_samples - j].sum()
                right_exchange_delta = insert_item_weight_old_dec * (insert_index) + weights_ori_right_1[: insert_index].sum()

                exchange_dec = exchange_dec + left_exchange_delta  - right_exchange_delta


                weights_ori_right = np.concatenate((weights_ori_right[:original_index], weights_ori_right[original_index+1:]), axis=None)
                weights_ori_left_1 = np.concatenate((weights_ori_left_1[:original_index], weights_ori_left_1[original_index+1:]), axis=None)
                weights_ori_left = np.concatenate((weights_ori_left[:insert_index], insert_item_weight_new_inc, weights_ori_left[insert_index:]), axis=None)
                weights_ori_right_1 = np.concatenate((weights_ori_right_1[:insert_index], insert_item_weight_new_dec, weights_ori_right_1[insert_index:]), axis=None)
                left_y_sorted.add(insert_item)

                left_size = j+1
                right_size -= 1
                if X_i_sorted[j] == X_i_sorted[j+1]:
                    continue
                k_inc = ((tot - (v_inc + u - t_inc)) - 2. * exchange_inc
                            ) / np.sqrt(tot - u) / np.sqrt(tot - v_inc)
                k_dec = ((tot - (v_dec + u - t_dec)) - 2. * exchange_dec
                            ) / np.sqrt(tot - u) / np.sqrt(tot - v_dec)

            if left_size < right_size and  k_dec > k_inc and left_size > low_bound or left_size > right_size and k_dec <= k_inc and right_size > low_bound:
                if left_size_pre_star == 0:
                    # Getting the left and right scores
                    # RHF score
                    left_score = [-1 * np.log(left_size / total_size)] * left_size
                    right_score = [-1 * np.log(right_size / total_size)] * right_size

                    # Concatenating the left and right
                    predict_concat = np.concatenate((left_score, right_score), axis=None)

                    # Calculating the kendall
                    tot_star = tot
                    # weigh joint ties
                    first = 0
                    t_star = 0
                    w = weights_ori[first]
                    s = w

                    for ii in range(1, n_samples):
                        if y_inc[first] != y_inc[ii] or predict_concat[xx[first]] != predict_concat[xx[ii]]:
                            t_star += s * (ii - first - 1)
                            first = ii
                            s = 0

                        w = weights_ori[ii]
                        s += w
                    t_star += s * (n_samples - first - 1)

                    # weigh ties in y_inc
                    first = 0
                    u_star = 0
                    w = weights_ori[first]
                    s = w
                    tie_star = {}
                    tie_l = 1
                    for ii in range(1, n_samples):
                        if y_inc[first] != y_inc[ii]:
                            if tie_l > 1:
                                tie_star[first] = tie_l
                            tie_l = 0

                            u_star += s * (ii - first - 1)
                            first = ii
                            s = 0

                        tie_l += 1
                        w = weights_ori[ii]
                        s += w

                    u_star += s * (n_samples - first - 1)
                    if first == 0: # x is constant (all ties)
                        u_star = np.nan

                    left_index = xx_rank[:left_size]
                    right_index = xx_rank[left_size:]

                    # weigh discordances
                    exchange_bi = 0
                    if left_size < right_size: # left_score > right_score
                        sum1 = sum([H_func(weights_bi, ind, N_data) for ind in left_index])
                        sum2 = weights_bi[left_index].sum()
                        exchange_bi = sum1 - (left_size-1)*sum2
                    else:
                        sum1 = sum([H_func(weights_bi, ind, N_data) for ind in right_index])
                        sum2 = weights_bi[right_index].sum()
                        exchange_bi = sum1 - (right_size-1)*sum2
                        bFirst = False

                    sum_v_2 =  weights_bi[right_index].sum()
                    v_star = sum_weights_bi*(left_size-1) + sum_v_2*(N_data-2*left_size) # v_star = weights_bi[left_index].sum()*(left_size-1) + weights_bi[right_index].sum()*(right_size-1)

                else:
                    ### check the exchange value
                    delta_right_index = xx_rank[left_size_pre_star:left_size]
                    N_shift = len(delta_right_index)

                    left_index = xx_rank[:left_size]
                    right_index = xx_rank[left_size:]

                    # calculate v_star by increments
                    del_v_1 = weights_bi[delta_right_index].sum()
                    del_v = sum_weights_bi*N_shift \
                            -2*N_shift*sum_v_2 \
                            -del_v_1*(N_data-2*left_size)

                    v_star +=  del_v
                    sum_v_2 -= del_v_1

                    if left_size < right_size: # left_score > right_score
                        sum2_add = del_v_1 # sum2_add = weights_bi[delta_right_index].sum()
                        delta1 = sum([H_func(weights_bi, ind, N_data) for ind in delta_right_index])
                        delta2 = (left_size-1)*sum2_add + \
                                N_shift * sum2
                        sum2 = sum2 + sum2_add
                        exchange_bi += (delta1-delta2)

                    else:# left_score < right_score
                        if bFirst == False:
                            sum2_add = del_v_1# sum([weights_bi[ind] for ind in delta_right_index])
                            delta1 = sum([H_func(weights_bi, ind, N_data) for ind in delta_right_index])
                            delta2 = (right_size-1)*sum2_add + \
                                    N_shift * sum2
                            sum2 = sum2 - sum2_add
                            exchange_bi -= (delta1-delta2)
                        else:
                            sum1 = sum([H_func(weights_bi, ind, N_data) for ind in right_index])
                            sum2 = weights_bi[right_index].sum()
                            exchange_bi = sum1 - (right_size-1)*sum2
                            bFirst = False
                kendall_split = ((tot_star - (u_star + v_star - t_star)) - 2. * exchange_bi) / np.sqrt(tot_star - u_star) / np.sqrt(tot_star - v_star)
                left_size_pre_star = left_size

                if kendall_split > kendall_base:
                    best_feature = feature
                    best_value = value
                    # Setting the best gain to the current one
                    kendall_base = kendall_split

            left_size_pre =  left_size
    return best_feature, best_value

class Node(object):
    """
    Node object
    """
    def __init__(self, y, X):
        super(Node, self).__init__()
        self.index = None
        self.depth = None
        self.data = X
        self.data_label = y
        self.type = 0 # 0, internal node; 1, leaf node
        self.parent = None
        self.impurity = 0
        self.label = -1 # -1, internal node; float, leaf node

        self.split_feature = None
        self.split_value = None

        self.left = None
        self.right = None

class Root(Node):
    """
    Node (Root) object
    """
    def __init__(self, y, X):
        self.depth = 0
        self.index = 0
        self.data = X
        self.data_label = y
        self.type = 0 # 0, internal node; 1, leaf node

class SizeBasedRegressionTree(object):
    """
    Regression Tree object

    :param int max_height: max height of the tree
    :param bool split_criterion: split criterion to use: 'kurtosis' or 'random'
    """
    def __init__(self, max_depth=20, max_nodes=128, epslon=0.1):
        super(SizeBasedRegressionTree, self).__init__()
        self.N = 0
        self.layer = 0
        self.criterion = 'kendall'
        self.max_depth = max_depth
        self.max_nodes = max_nodes
        self.reg_scores = None
        self.queue1 = collections.deque()
        self.leaf_nodes = [] # leaf node index
        self.all_nodes = [] # all node index
        self.leaf_gt = [] # leaf node gt
        self.epslon = epslon


    def generate_node(self, y, X, depth=None, parent=None):
        self.N += 1

        node = Node(y, X)
        node.depth = depth
        node.index = self.N
        node.parent = parent
        self.all_nodes.append(node.index)

        return node

    def set_leaf(self, node, y, X):
        """
        Transforms generic node into leaf

        :param node: generic node to transform into leaf
        :param data: node data used to define node size and data indexes corresponding to node
        """
        node.type = 1
        node.data = X
        self.leaf_nodes.append(node.index)
        self.leaf_gt.append(y.index)
        
        node.label = 1/len(y)

        self.reg_scores[node.data.index] = node.label

    def build(self, low_bound_layer):
        """
        Function which recursively builds the tree

        :param node: current node
        :param data: data corresponding to current node
        """
        self.queue2 = collections.deque()

        low_bound = low_bound_layer
        leaf_flag = 1

        while(self.queue1):
            currentnode = self.queue1.popleft()

            X = currentnode.data
            y = currentnode.data_label

            if X.shape[0] == 0:
                os.sys.exit('Empty dataset!')
                return
            if X.shape[0] == 1 :
                self.set_leaf(currentnode, y, X)
                low_bound_layer = X.shape[0]
                low_bound = X.shape[0]
                continue
            if X.duplicated().sum() == X.shape[0] - 1:
                self.set_leaf(currentnode, y, X)
                continue
            if currentnode.depth >= self.max_depth or len(self.all_nodes) >= self.max_nodes:
                self.set_leaf(currentnode, y, X)
                continue


            best_feature, best_value = get_best_split(y, X, low_bound, self.total_size, self.epslon)

            if best_feature is not None:
                y_left, X_left = y[X[best_feature] < best_value], X[X[best_feature] < best_value]
                y_right, X_right = y[X[best_feature] >= best_value], X[X[best_feature] >= best_value]
                currentnode.left =  self.generate_node(y_left, X_left, depth = currentnode.depth+1, parent = currentnode)
                currentnode.right = self.generate_node(y_right, X_right, depth = currentnode.depth+1, parent = currentnode)

                currentnode.split_feature = best_feature
                currentnode.split_value = best_value
                if len(X_left) == 0 or len(X_right) == 0:
                    self.queue2.append(currentnode)
                else:
                    if len(X_left) < len(X_right):
                        self.queue2.append(currentnode.left)
                        self.queue2.append(currentnode.right)
                        low_bound = X_right.shape[0]
                    else:
                        self.queue2.append(currentnode.right)
                        self.queue2.append(currentnode.left)
                        low_bound = X_left.shape[0]
                leaf_flag = 0


            elif leaf_flag == 1:
                self.set_leaf(currentnode, y, X)
                low_bound_layer = X.shape[0]
                low_bound = X.shape[0]
                # print('leaf: %d, %.2f, %.2f'%(len(X), currentnode.label, currentnode.data_label.max()))
            else:
                self.queue2.append(currentnode)
                low_bound = X.shape[0]

        self.layer += 1
        if len(self.queue2) > 0:
            self.queue1 = self.queue2.copy()
            self.build(low_bound_layer)
        else:
            return

    def fit(self, y, X):
        """
        Build tree function: generates the root node and successively builds the tree recursively

        :param data: the dataset
        """
        # Transform y to pandas.Series
        y = pd.Series(y)
        self.total_size = len(y)
        self.n_features_in_ = X.shape[1]
        self.tree_ = Root(y, X)

        self.reg_scores = np.zeros(X.shape[0])

        self.queue1.append(self.tree_)

        self.build(0)

    def predict(self, mySample):
        current_node = self.tree_
        while (current_node.type == 0):
            splitFeature = current_node.split_feature
            splitValue = current_node.split_value
            if (mySample[splitFeature] <= splitValue):
                current_node = current_node.left
            else:
                current_node = current_node.right
        if current_node.label == -1:
            print('error!')
            import ipdb;ipdb.set_trace()

            return None
        else:
            return current_node.label
    def predict_train(self):
        return self.reg_scores

    def traverse_tree(self, root, attribute_list=[]):
        current_node = []
        ###
        # node / node type (LN - leave node, IN - internal node) left child / right child / feature / threshold / node_depth / majority class (starts with index 0)
        ###
        # node id
        current_node.append(root.index)
        # node type
        if root.type == 1:
            current_node.append('LN')
            # left child / right child / feature / threshold /
            current_node.extend([-1,-1,-1,-1])
            # node depth
            current_node.append(root.depth)
            # label
            current_node.append(root.label)
            # n_node_samples
            current_node.append(root.data.shape[0])

            attribute_list.append(current_node)
            return attribute_list

        else:
            current_node.append('IN')
            # left child / right child / feature / threshold /
            current_node.extend([root.left.index, root.right.index, root.split_feature, root.split_value])
            # node depth
            current_node.append(root.depth)
            # label
            current_node.append(-1)
            # n_node_samples
            current_node.append(root.data.shape[0])
            attribute_list.append(current_node)

            if root.left is not None:
                self.traverse_tree(root.left, attribute_list)
            if root.right is not None:
                self.traverse_tree(root.right, attribute_list)
            return attribute_list
    def get_attribute(self):
        attribute_list = []

        attribute_list = self.traverse_tree(self.tree_, attribute_list)
        attribute_list = np.array(attribute_list)
        node_ids = attribute_list[:, 0]
        self.node_count = len(node_ids)
        node_ids = [int(x) for x in node_ids]
        node_ids_sorted = np.argsort(node_ids)
        attribute_list_sorted = attribute_list[node_ids_sorted]
        # extract attribute columns
        self.tree_.children_left = [int(x) for x in attribute_list_sorted[:, 2]]
        self.tree_.children_right = [int(x) for x in attribute_list_sorted[:, 3]]
        self.tree_.n_outputs = 1

        # self.is_leaf = [1 if x == 'LN' else 0 for x in attribute_list_sorted[:, 1]]
        self.tree_.value = [-1 if x == '-1' else float(x) for x in attribute_list_sorted[:, 7]]
        self.tree_.value = np.reshape(self.tree_.value, (self.node_count, 1, -1))
        self.tree_.feature = [int(x) for x in attribute_list_sorted[:, 4]]
        self.tree_.threshold = [-1 if x == '-1' else float(x) for x in attribute_list_sorted[:, 5]]
        self.tree_.impurity = np.full((self.node_count), -1)
        self.tree_.n_node_samples = [int(x) for x in attribute_list_sorted[:, 8]]
        self.tree_.weighted_n_node_samples = self.tree_.n_node_samples
        self.tree_.n_classes = [1]

if __name__ == '__main__':
    print(Harmonic_n)
