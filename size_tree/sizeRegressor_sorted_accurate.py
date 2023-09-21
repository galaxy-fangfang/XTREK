"""Main module."""
import collections
from gc import collect
import os
from scipy.stats import kurtosis
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.metrics import average_precision_score, mean_squared_error
from rhf import RHF
import math
from math import log

def H_correct(n):
    # gamma = 0.57721566490153286060651209008240243104215933593992
    # return gamma + log(n) + 0.5/n - 1./(12*n**2) + 1./(120*n**4)
    return sum(1/d for d in range(1,n+1))

Harmonic_n = np.array([H_correct(n) for n in range(1,101+1)])

def H(n):
    # Euler-Mascheroni constant
    gamma = 0.57721566490153286060651209008240243104215933593992
    if n > 500:
        return gamma + log(n)
    elif n > 100:
        return gamma + log(n) + 0.5/n
    elif n > 0:
        return Harmonic_n[n-1]
    else:
        return 0

def H_func(w, n, N):
    y = H(N - n) + (N-n-2)*w[n]
    return y

def weighted_kendall(rhf_scores, score_per_tree):
    tau, p_value = stats.weightedtau(rhf_scores, score_per_tree, rank=None)
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
        # cdef intp_t n, i, j, k
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
        # cdef intp_t length0, length1, middle, i, j, k
        # cdef float64_t weight, residual

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
        # cdef intp_t length0, length1, middle, i, j, k
        # cdef float64_t weight, residual

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
    # print("tot: %.3f, u: %.3f, v: %.3f, t: %.3f, swap: %.3f"%(tot, u, v, t, exchanges_weight[0]))
    # print(tau)
    # weights_y = weigher(rank[perm])
    return min(1., max(-1., tau)), tot, u, v, t, exchanges_weight[0], weights_x, tot - (v + u - t) - 2. * exchanges_weight[0], np.sqrt(tot - u) * np.sqrt(tot - v)


def get_mean_square_error_split_optimal(y, X):
    """
    Get attribute split according to Mean Square Error

    :param y: the label of the node
    :param X: the dataset of the node
    :returns:
        - feature_index: the attribute index to split
        - feature_split: the attribute value to split

    The MSE proxy is derived from
            sum_{i left}(y_i - y_pred_L)^2 + sum_{i right}(y_i - y_pred_R)^2
            = sum(y_i^2) - n_L * mean_{i left}(y_i)^2 - n_R * mean_{i right}(y_i)^2
        Neglecting constant terms, this gives:
            - 1/n_L * sum_{i left}(y_i)^2 - 1/n_R * sum_{i right}(y_i)^2
    """
    # turn dataframe to array
    X = X.values
    y = y.values


    d = X.shape[1]
    n_samples = X.shape[0]

    # impurity_improvement_base = -1 * np.inf
    impurity_improvement_base = 0
    sum_total = np.sum(y)
    sqre_sum_total = np.sum(y**2)
    y_mean = np.mean(y)

    best_feature = None
    best_value = None

    for i in range(d):

        feature = i

        sum_left = 0
        sum_right = sum_total
        X_i_sorted = sorted(X[:, feature])
        X_i_sorted_index = np.argsort(X[:, feature])
        y_i_sorted = y[X_i_sorted_index]
        n_splits_unique = len(np.unique(X[:, i])) - 1
        if n_splits_unique <= 0:
            continue

        for j in range(len(X_i_sorted)):
            # current_ind = i + j
            if j + 1 >= n_samples:
                break

            sum_left += y_i_sorted[j]
            sum_right  = sum_total - sum_left
            if X_i_sorted[j] == X_i_sorted[j+1]:
                continue

            value = X_i_sorted[j] / 2.0 + X_i_sorted[j+1] / 2.0
            # Calculating the mse impurity
            impurity_improvement_current = sum_left * sum_left / (j + 1) + sum_right * sum_right /(n_samples - j - 1)# - n_samples * y_mean**2
            mse_split = sqre_sum_total - impurity_improvement_current

            ### impurity_improvement_current ==
            ### aa = np.sum((y-y_mean)**2) - np.sum((y_i_sorted[:j+1] - np.mean(y_i_sorted[:j+1]))**2) - np.sum((y_i_sorted[j+1:]-np.mean(y_i_sorted[j+1:]))**2)
            # import ipdb;ipdb.set_trace()

            # Checking if this is the best split so far
            if impurity_improvement_current >  impurity_improvement_base:
                best_feature = feature
                best_value = value

                # Setting the best gain to the current one
                impurity_improvement_base = impurity_improvement_current

    return best_feature, best_value

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

def get_rhf_split(y, X, low_bound, total_size, epslon=0.1, select_way='k3'):
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
        right_size = N_data

        left_size_pre = 0
        left_size_pre_star = 0
        bFirst = True

        tot, u = None, None

        from sortedcontainers import SortedList
        left_y_sorted = SortedList()
        right_y_sorted = SortedList(y_i_sorted_by_f)
        from collections import deque
        right_y = deque(y_i_sorted_by_f)
        original_index_pre = 0
        for j in range(len(X_i_sorted)):
            if j + 1 >= n_samples:
                break
            value = X_i_sorted[j] / 2.0 + X_i_sorted[j+1] / 2.0

            step = math.ceil(n_samples*epslon)
            if step==0 or j % step == 0:
                left_size_pre = 0
            if left_size_pre== 0:
                # sorted container
                insert_item = right_y.popleft()
                left_y_sorted.add(insert_item)
                original_index_pre = right_y_sorted.index(insert_item)                
                _ = right_y_sorted.pop(original_index_pre)

                if X_i_sorted[j] == X_i_sorted[j+1]:
                    continue
                left_size = j + 1
                right_size  = n_samples - left_size
                if left_size <= low_bound:
                    continue
                if right_size <= low_bound:
                    break

                # Getting the left and right ys
                # left_y = y_i_sorted_by_f[ :left_size]
                

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

                k_inc, tot, u, v_inc, t_inc, exchange_inc, weights_rhf, con_minus_dis_inc, n_minus_tie_inc = weighted_kendall_my(y_inc, concat_inc, left_size, tot=tot, u=u)
                k_dec, tot, u, v_dec, t_dec, exchange_dec, weights_rhf, con_minus_dis_dec, n_minus_tie_dec = weighted_kendall_my(y_inc, concat_dec, right_size, tot=tot, u=u)

                # print('===%d, begin: %.3f, %.3f' % (i, exchange_inc, exchange_dec))
                # print('n: ', n_samples, 'i: ', i, 'j: ', j)#, 'k_inc: %.4f'% k_inc,  'k_dec: %.4f'% k_dec, 'left: %.2f, %.2f '%( np.max(left_y_sorted),  np.max(right_y_sorted)))
            else:

                insert_item = right_y.popleft()
                insert_index = left_y_sorted.bisect_right(insert_item)
                original_index = right_y_sorted.index(insert_item)
                _ = right_y_sorted.pop(original_index)

                #exchange for left to right
                mv = j
                # insert_item_weight_old_inc = weights_rhf[mv+original_index]
                # insert_item_weight_new_inc = weights_rhf[insert_index]

                # delta_inc = (insert_item_weight_new_inc*(mv - insert_index)
                #             - insert_item_weight_new_inc*original_index
                #             - insert_item_weight_old_inc*original_index
                #             + insert_item_weight_old_inc*(mv - insert_index)
                #             +(H(N_data - insert_index - 1) - H(N_data - mv - 1))
                #             -(H(N_data - mv -1) - H(N_data - mv - original_index - 1))
                #             -(H(N_data - mv) - H(N_data - mv - original_index))
                #             +(H(N_data - insert_index) - H(N_data - mv)))


                delta_inc = (
                    # 2 * (weights_rhf[mv] * (mv-insert_index) + weights_rhf[insert_index: mv].sum())
                    # -2 * (weights_rhf[mv] * original_index + weights_rhf[mv+1:mv+original_index+1].sum())
                    2 * (weights_rhf[mv] * (mv-insert_index) + H(N_data-insert_index)-H(N_data-mv))
                    -2 * (weights_rhf[mv] * original_index + H(N_data-mv-1)-H(N_data-mv-original_index-1))
                    )

                # delta_exchange_inc = (
                #             + insert_item_weight_new_inc*original_index
                #             +(H(N_data - mv -1) - H(N_data - mv - original_index - 1))
                #             - insert_item_weight_old_inc*(mv - insert_index)
                #             - (H(N_data - insert_index) - H(N_data - mv)))
                # delta_exchange_inc = (
                #     -(weights_rhf[mv] * (mv-insert_index) + weights_rhf[insert_index: mv].sum())
                #     +(weights_rhf[mv] * original_index + weights_rhf[mv+1:mv+original_index+1].sum())
                # )

                con_minus_dis_inc = con_minus_dis_inc + delta_inc
                # exchange_inc += delta_exchange_inc
                
                #exchange for right to left
                mu = N_data-mv
                ### mauro
                # insert_item_weight_new_dec = weights_rhf[mu+insert_index-1]
                # insert_item_weight_old_dec = weights_rhf[original_index]
                # delta_dec = (
                #     insert_item_weight_new_dec * (insert_index)
                #     - insert_item_weight_new_dec * (mu - original_index - 1)
                #     - insert_item_weight_old_dec * (mu - original_index - 1)
                #     + insert_item_weight_old_dec * (insert_index)
                #     + (H(N_data - mu + 1) - H(N_data - mu -insert_index + 1))
                #     - (H(N_data - original_index) - H(N_data - mu + 1))
                #     - (H(N_data - original_index - 1) - H(N_data - mu))
                #     + (H(N_data - mu) - H(N_data - mu - insert_index))
                # )

                delta_dec = (
                    # 2 * (weights_rhf[mu-1] * insert_index + weights_rhf[mu:mu+insert_index].sum())
                    # -2 *(weights_rhf[mu-1] * (mu-original_index-1) + weights_rhf[original_index:mu-1].sum())
                    2 * (weights_rhf[mu-1] * insert_index + H(N_data-mu)-H(N_data-mu-insert_index))
                    -2 *(weights_rhf[mu-1] * (mu-original_index-1) + H(N_data-original_index)-H(N_data-mu+1))
                )
                
                # delta_exchange_dec = (
                #     + (H(N_data - original_index) - H(N_data - mu + 1))
                #     + (insert_item_weight_new_dec * (mu - original_index - 1) )
                #     - (insert_item_weight_old_dec * (insert_index) )
                #     - (H(N_data - mu) - H(N_data - mu - insert_index))
                # )
                # delta_exchange_dec = (
                #     -(weights_rhf[mu-1]*insert_index+weights_rhf[mu:mu+insert_index].sum())
                #     +(weights_rhf[mu-1]*(mu-1-original_index) + weights_rhf[original_index:mu-1].sum())
                # )
                con_minus_dis_dec = con_minus_dis_dec + delta_dec
                # exchange_dec += delta_exchange_dec

                left_y_sorted.add(insert_item)

                left_size = j+1
                right_size -= 1

                if X_i_sorted[j] == X_i_sorted[j+1]:
                    continue
                k_inc = con_minus_dis_inc / n_minus_tie_inc
                k_dec = con_minus_dis_dec / n_minus_tie_dec
                
            if left_size < right_size and  k_dec > k_inc and left_size > low_bound or left_size > right_size and k_dec <= k_inc and right_size > low_bound:
                if select_way == 'k3' or select_way == 'mix':
                    if left_size_pre_star == 0:
                        # Getting the left and right scores
                        # RHF score
                        left_score = [-1 * np.log(left_size / total_size)] * left_size
                        right_score = [-1 * np.log(right_size / total_size)] * right_size

                        # Concatenating the left and right
                        predict_concat = np.concatenate((left_score, right_score), axis=None)

                        # Calculating the kendall
                        # kendall_split = weighted_kendall(y_inc, predict_concat[xx])
                        # kendall_split1, tot1, u1, v1, t1, exchange1, _ = weighted_kendall_my(y_inc, predict_concat[xx], left_size, y_sorted=False)

                        tot_star = tot
                        # weigh joint ties
                        first = 0
                        t_star = 0
                        w = weights_rhf[first]
                        s = w

                        for ii in range(1, n_samples):
                            if y_inc[first] != y_inc[ii] or predict_concat[xx[first]] != predict_concat[xx[ii]]:
                                t_star += s * (ii - first - 1)
                                first = ii
                                s = 0

                            w = weights_rhf[ii]
                            s += w
                        t_star += s * (n_samples - first - 1)

                        # weigh ties in y_inc
                        first = 0
                        u_star = 0
                        w = weights_rhf[first]
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
                            w = weights_rhf[ii]
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

                        # print('exchange1: %.3f, exchangestar: %.3f, exchange_bi: %.3f, ' % (exchange1, exchanges_star, exchange_bi))
                        # weigh ties in predict_concat[xx]
                        sum_v_2 =  weights_bi[right_index].sum()
                        v_star = sum_weights_bi*(left_size-1) + sum_v_2*(N_data-2*left_size) # v_star = weights_bi[left_index].sum()*(left_size-1) + weights_bi[right_index].sum()*(right_size-1)

                    else:
                        ### check the exchange value
                        # left_score = [-1 * np.log(left_size / total_size)] * left_size
                        # right_score = [-1 * np.log(right_size / total_size)] * right_size
                        # predict_concat = np.concatenate((left_score, right_score), axis=None)
                        # kendall_split1, tot1, u1, v1, t1, exchange1, _ = weighted_kendall_my(y_inc, predict_concat[xx], left_size, y_sorted=False)

                        delta_right_index = xx_rank[left_size_pre_star:left_size]
                        N_shift = len(delta_right_index)

                        # # weigh joint ties
                        # first = 0
                        # t_star = 0
                        # w = weights_rhf[first]
                        # s = w

                        # for ii in range(1, n_samples):
                        #     if y_inc[first] != y_inc[ii] or predict_concat[xx[first]] != predict_concat[xx[ii]]:
                        #         t_star += s * (ii - first - 1)
                        #         first = ii
                        #         s = 0

                        #     w = weights_rhf[ii]
                        #     s += w
                        # t_star += s * (n_samples - first - 1)

                        left_index = xx_rank[:left_size]
                        right_index = xx_rank[left_size:]

                        # calculate v_star by increments
                        # v_star_ref = weights_bi[left_index].sum()*(left_size-1) + weights_bi[right_index].sum()*(right_size-1)
                        del_v_1 = weights_bi[delta_right_index].sum()
                        del_v = sum_weights_bi*N_shift \
                                -2*N_shift*sum_v_2 \
                                -del_v_1*(N_data-2*left_size)

                        v_star +=  del_v
                        sum_v_2 -= del_v_1
                        # print(v_star,  del_v, v_star-v_star_ref)

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
                    # print('exchange1: %.3f, exchange_bi: %.3f, delta: %0.3f' % (exchange1,  exchange_bi, exchange_bi-exchange1))
                    if select_way == 'mix':
                        kendall_split += max(k_inc, k_dec)

                elif select_way == 'max':
                    kendall_split = max(k_inc, k_dec)


                if kendall_split > kendall_base:
                    best_feature = feature
                    best_value = value
                    # Setting the best gain to the current one
                    kendall_base = kendall_split

                ###
            #     if DEBUG:
            #         if feature == 0:
            #             if value >0. and value < 1:
            #                 print('i: %d, j: %d, value: %.3f, k: %.4f, k_inc: %.4f, k_dec: %.4f' % (feature, left_size, value, kendall_split, k_inc, k_dec))
            #                 print('inc ori, dec ori: %.3f, %.3f' %(weighted_kendall(y_inc, concat_inc), weighted_kendall(y_inc, concat_dec)))
            #             if value >22176. and value < 22178.:
            #                 print('i: %d, j: %d, value: %.3f, k: %.4f, k_inc: %.4f, k_dec: %.4f' % (feature, left_size, value, kendall_split, k_inc, k_dec))
            #                 print('inc ori, dec ori: %.3f, %.3f' %(weighted_kendall(y_inc, concat_inc), weighted_kendall(y_inc, concat_dec)))

            # if DEBUG:
            #     if feature == 1:
            #         if value >30000. and value < 40000:
            #             left_score = [-1 * np.log(left_size / total_size)] * left_size
            #             right_score = [-1 * np.log(right_size / total_size)] * right_size

            #             # Concatenating the left and right
            #             predict_concat = np.concatenate((left_score, right_score), axis=None)

            #             # Calculating the kendall
            #             k3 = weighted_kendall(y_inc, predict_concat[xx])
            #             print('i: %d, j: %d, value: %.3f, k: %.4f, k_inc: %.4f, k_dec: %.4f' % (feature, left_size, value, k3, k_inc, k_dec))
            #             print('inc ori, dec ori: %.3f, %.3f' %(weighted_kendall(y_inc, concat_inc), weighted_kendall(y_inc, concat_dec)))
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
        # self.data_index = None
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
    def __init__(self, max_depth=None, max_nodes=128, score_way='RHF', epslon=0.1, select_way='k3'):
        super(SizeBasedRegressionTree, self).__init__()
        self.N = 0
        self.layer = 0
        self.criterion = 'kendall'
        self.max_depth = max_depth
        self.max_nodes = max_nodes
        self.score_way = score_way
        self.select_way = select_way
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
        if self.score_way in ['RHF', 'MEANRHF'] :
            p =  len(y)/ self.total_size
            if p == 0:
                node.label = 0
            else:
                node.label = np.log(1 / (p))*10
                node.label = 1/len(y)
        elif self.score_way == 'MEAN':
            node.label = np.mean(y)
        self.reg_scores[node.data.index] = node.label


    def build_mean(self, node, y, X):
        """
        Function which recursively builds the tree

        :param node: current node
        :param data: data corresponding to current node
        """
        if X.shape[0] == 0:
            os.sys.exit('Empty dataset!')
            return
        if X.shape[0] <= 1 :
            self.set_leaf(node, y, X)
            return
        if X.duplicated().sum() == X.shape[0] - 1:
            self.set_leaf(node, y, X)
            return
        if node.depth >= self.max_depth:
            self.set_leaf(node, y, X)
            return
        if node.depth == 0:
            node.impurity = np.sum((y - np.mean(y)) ** 2) / len(y)

        node.impurity = np.sum((y - np.mean(y)) ** 2) / len(y)
        # best_feature, best_value = get_rhf_split(y, X)
        best_feature, best_value = get_mean_square_error_split_optimal(y, X)

        if best_feature is not None:
            y_left, X_left = y[X[best_feature] < best_value], X[X[best_feature] < best_value]
            y_right, X_right = y[X[best_feature] >= best_value], X[X[best_feature] >= best_value]
            node.left =  self.generate_node(y_left, X_left, depth = node.depth+1, parent = node)
            node.right = self.generate_node(y_right, X_right, depth = node.depth+1, parent = node)

            node.split_feature = best_feature
            node.split_value = best_value
            if X_left.shape[0] == 0 or X_right.shape[0] == 0:
                self.set_leaf(node, y, X)
                return
            else:
                self.build_mean(node.left, y_left, X_left, node)
                self.build_mean(node.right, y_right, X_right, node)
        else:
            self.set_leaf(node, y, X)
            return

    def build(self, low_bound_layer):
        """
        Function which recursively builds the tree

        :param node: current node
        :param data: data corresponding to current node
        """
        self.queue2 = collections.deque()
        # print('layer: ', self.layer, ' nodes: ', len(self.queue1))

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
                # print('leaf: %d, %.2f, %.2f'%(len(X), currentnode.label, currentnode.data_label.max()))
                continue
            if X.duplicated().sum() == X.shape[0] - 1:
                self.set_leaf(currentnode, y, X)
                # print('leaf: %d, %.2f, %.2f'%(len(X), currentnode.label, currentnode.data_label.max()))
                continue
            if currentnode.depth >= self.max_depth or len(self.all_nodes) >= self.max_nodes:
                self.set_leaf(currentnode, y, X)
                # print('leaf: %d, %.2f, %.2f'%(len(X), currentnode.label, currentnode.data_label.max()))
                continue


            best_feature, best_value = get_rhf_split(y, X, low_bound, self.total_size, self.epslon, self.select_way)

            if best_feature is not None:
                y_left, X_left = y[X[best_feature] < best_value], X[X[best_feature] < best_value]
                y_right, X_right = y[X[best_feature] >= best_value], X[X[best_feature] >= best_value]
                # print('best_feature, best_value, leftsize, rightsize: %d, %f, %d, %d' %
                # (best_feature, best_value, len(y_left), len(y_right)))
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

    def build1(self, low_bound_layer):
        """
        Function which recursively builds the tree

        :param node: current node
        :param data: data corresponding to current node
        """
        self.queue2 = collections.deque()
        # print('layer: ', self.layer, ' nodes: ', len(self.queue1))

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
                print('leaf: %d, %.2f, %.2f'%(len(X), currentnode.label, currentnode.data_label.max()))
                continue
            if X.duplicated().sum() == X.shape[0] - 1:
                self.set_leaf(currentnode, y, X)
                print('leaf: %d, %.2f, %.2f'%(len(X), currentnode.label, currentnode.data_label.max()))
                continue
            if currentnode.depth >= self.max_depth:
                self.set_leaf(currentnode, y, X)
                print('leaf: %d, %.2f, %.2f'%(len(X), currentnode.label, currentnode.data_label.max()))
                continue
            if currentnode.depth <= 3:
                best_feature, best_value = get_mean_square_error_split_optimal(y, X)
            else:
                best_feature, best_value = get_rhf_split(y, X, low_bound, self.total_size, self.epslon, self.select_way)

            if best_feature is not None:
                y_left, X_left = y[X[best_feature] < best_value], X[X[best_feature] < best_value]
                y_right, X_right = y[X[best_feature] >= best_value], X[X[best_feature] >= best_value]
                print('depth, best_feature, best_value, leftsize, rightsize: %d, %d, %f, %d, %d' %
                (currentnode.depth, best_feature, best_value, len(y_left), len(y_right)))
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
                print('leaf: %d, %.2f, %.2f'%(len(X), currentnode.label, currentnode.data_label.max()))
            else:
                self.queue2.append(currentnode)
                low_bound = X.shape[0]

        self.layer += 1
        if len(self.queue2) > 0:
            self.queue1 = self.queue2.copy()
            self.build1(low_bound_layer)
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
        # self.total_size_unique = y.nunique()
        self.tree_ = Root(y, X)
        # self.tree_.label, self.tree_.data = y, X

        self.reg_scores = np.zeros(X.shape[0])

        self.queue1.append(self.tree_)

        if self.score_way == 'MEAN':
            self.build_mean(self.tree_, y, X)
        elif self.score_way == 'RHF':
            self.build(0)
        elif self.score_way == 'MEANRHF':
            self.build1(0)#the first 3 layers: MSE, tthe rest: kendall

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
    dataset = 'breastcancer'
    dataset_path = os.path.join('../datasets/klf/', dataset)
    df = pd.read_csv(dataset_path, header=0)
    # df = df.sample(frac=1).reset_index(drop=True)
    y = df["label"]
    sample_size = y.value_counts()[1]
    X = df.drop("label", axis=1)
    X.columns = [int(x) for x in X.columns]
    print('Dataset: %s ' % dataset, 'n=', len(y), ' d=', len(X.columns), ' anomaly=', sample_size)

    ### RHF in CYTHON VERSION
    black_box = RHF(num_trees=100, max_height=5, seed_state=10010, check_duplicates=True, decremental=False,
                    use_kurtosis=True)
    black_box.fit(X)
    rhf_scores, scores_all = black_box.get_scores()
    rhf_scores = rhf_scores / 100
    rhf_ap = average_precision_score(y, rhf_scores)

    import time
    starttime = time.time()
    explainer = SizeBasedRegressionTree(max_depth=10, score_way='RHF', epslon=1)
    explainer.fit(rhf_scores, X)
    print('time: %.3f'%(time.time()-starttime))
    # explainer_predict = X.apply(explainer.predict, axis=1)
    explainer_predict = explainer.predict_train()
    explainer_ap = average_precision_score(y, explainer_predict)
    print('RHF AP: %.4f ' % (rhf_ap))
    print('size AP: %.4f ' % explainer_ap)#, 'mse: %f' % mean_squared_error(rhf_scores, explainer_predict))

    ### test binary search
    # arr = [1,3,4,6]
    # index = binarySearch(arr, 0, 3, 0.5)
    # print(index)
    ### test harmonic number
    # print(Harmonic_n)
