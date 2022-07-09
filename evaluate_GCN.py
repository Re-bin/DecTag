import numpy as np
import torch
import time
import torch.nn as nn
import math
import random
import model
import data_util

def minibatch(*tensors, **kwargs):
    batch_size = kwargs.get('batch_size')
    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)

def getLabel(test_data, pred_data):
    r = []
    for i in range(len(test_data)):
        groundTrue = test_data[i]
        predictTopK = pred_data[i]
        pred = list(map(lambda x: x in groundTrue, predictTopK))
        pred = np.array(pred).astype("float")
        r.append(pred)
    return np.array(r).astype('float')

def Recall_ATk(test_data, r, k):
    right_pred = r[:, :k].sum(1)
    recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])
    recall = np.sum(right_pred/recall_n)
    return {'recall': recall}

def NDCGatK_r(test_data, r, k):
    assert len(r) == len(test_data)
    pred_data = r[:, :k]

    test_matrix = np.zeros((len(pred_data), k))
    for i, items in enumerate(test_data):
        length = k if k <= len(items) else len(items)
        test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = np.sum(max_r * 1./np.log2(np.arange(2, k + 2)), axis=1)
    dcg = pred_data*(1./np.log2(np.arange(2, k + 2)))
    dcg = np.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg/idcg
    ndcg[np.isnan(ndcg)] = 0.
    return np.sum(ndcg)

def test_one_batch(X, topks):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    r = getLabel(groundTrue, sorted_items)
    pre, recall, ndcg = [], [], []
    for k in topks:
        ret = Recall_ATk(groundTrue, r, k)
        recall.append(ret['recall'])
        ndcg.append(NDCGatK_r(groundTrue, r, k))
    return {'recall':np.array(recall),
            'ndcg':np.array(ndcg)}

def GCN_test(dataset, model, split_index, test_batch_size, topks):
    dataset : data_util.GCNData
    test_dict = dataset.test_dict
    model.set_model_type('test')
    model = model.eval()
    max_k = max(topks)
    results = {'recall': np.zeros(len(topks)),
               'ndcg': np.zeros(len(topks))}
    with torch.no_grad():
        items = list(dataset.test_dict.keys())
        items_list = []
        rating_list = []
        ground_true_list = []
        total_batch = len(items) // test_batch_size + 1
        for batch_items in minibatch(items, batch_size=test_batch_size):
            ground_true = [test_dict[item] for item in batch_items]
            batch_items_gpu = torch.Tensor(batch_items).long().cuda()
            rating = model.get_user_rating(batch_items_gpu)
            # all the videos in test set do not exist in training set
            # no need to drop tags
            _, rating_K = torch.topk(rating, k=max_k)
            rating = rating.cpu().numpy()
            del rating
            items_list.append(batch_items)
            rating_list.append(rating_K.cpu())
            ground_true_list.append(ground_true)
        assert total_batch == len(items_list)
        X = zip(rating_list, ground_true_list)
        pre_results = []
        for x in X:
            pre_results.append(test_one_batch(x, topks))
        for result in pre_results:
            results['recall'] += result['recall']
            results['ndcg'] += result['ndcg']
        results['recall'] /= float(len(items))
        results['ndcg'] /= float(len(items))

        print('----' * 18)
        for i in range(len(topks)):
            print("Split{} Testing Set: Recall@{}: {:.5f} NDCG@{}: {:.5f}".format(split_index, topks[i], results['recall'][i], topks[i], results['ndcg'][i], topks[i]))
        print('----' * 18)

        return results

def GCN_valid(dataset, model, epoch, start_time, split_index, valid_batch_size, topks):
    dataset: data_util.GCNData
    valid_dict = dataset.valid_dict
    model.set_model_type('valid')
    model = model.eval()
    max_k = max(topks)
    results = {'recall': np.zeros(len(topks))}
    with torch.no_grad():
        items = list(dataset.valid_dict.keys())
        items_list = []
        rating_list = []
        ground_true_list = []
        total_batch = len(items) // valid_batch_size + 1
        for batch_items in minibatch(items, batch_size=valid_batch_size):
            ground_true = [valid_dict[item] for item in batch_items]
            batch_items_gpu = torch.Tensor(batch_items).long().cuda()
            rating = model.get_user_rating(batch_items_gpu)
            # all the videos in test set do not exist in training set
            # no need to drop tags
            _, rating_K = torch.topk(rating, k=max_k)
            rating = rating.cpu().numpy()
            del rating
            items_list.append(batch_items)
            rating_list.append(rating_K.cpu())
            ground_true_list.append(ground_true)
        assert total_batch == len(items_list)
        X = zip(rating_list, ground_true_list)
        pre_results = []
        for x in X:
            pre_results.append(test_one_batch(x, topks))
        for result in pre_results:
            results['recall'] += result['recall']
        results['recall'] /= float(len(items))

        print('----' * 18)
        print("Runing Epoch {:03d} ".format(epoch) + "costs " + time.strftime(
            "%H: %M: %S", time.gmtime(time.time() - start_time)))
        print("Split{} Validating Set: Recall@{}: {:.5f}".format(split_index, topks[0], results['recall'][0]))

    return results['recall'][0]
