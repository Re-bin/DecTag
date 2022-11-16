import numpy as np
import torch
import time
import torch.nn as nn
import math
import random


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
    recall = np.sum(right_pred / recall_n)
    return {'recall': recall}


def NDCGatK_r(test_data, r, k):
    assert len(r) == len(test_data)
    pred_data = r[:, :k]

    test_matrix = np.zeros((len(pred_data), k))
    for i, items in enumerate(test_data):
        length = k if k <= len(items) else len(items)
        test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = np.sum(max_r * 1. / np.log2(np.arange(2, k + 2)), axis=1)
    dcg = pred_data * (1. / np.log2(np.arange(2, k + 2)))
    dcg = np.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg / idcg
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
    return {'recall': np.array(recall),
            'precision': np.array(pre),
            'ndcg': np.array(ndcg)}


def get_metrics(model, item_tags, video_features, itemid_videohiddenid_map, tag_num, topks, split_index, batch_size=10240):
    items = list(item_tags.keys())
    result_matrix = np.zeros(shape=(len(items), tag_num))
    steps = int(len(items) // batch_size + 1)
    batchs_item_feature = []
    for i in range(steps):
        start_pos = i * batch_size
        end_pos = int(min((i + 1) * batch_size, len(items)))
        if start_pos < len(items):
            batch_items = items[start_pos: end_pos]
            batch_items_feature = []
            for item in batch_items:
                batch_items_feature.append(video_features[itemid_videohiddenid_map[item]])
            batch_items_feature = torch.FloatTensor(batch_items_feature).cuda()
            batchs_item_feature.append(batch_items_feature)
    for i in range(tag_num):
        one_tag_predictions = []
        for j in range(len(batchs_item_feature)):
            batch_items_feature = batchs_item_feature[j]
            this_batch_size = len(batch_items_feature)
            tag_id = torch.LongTensor([i]).repeat(this_batch_size).view(this_batch_size, 1).cuda()
            category_id = torch.LongTensor([0, 1]).repeat(this_batch_size).view(this_batch_size, 2).cuda()
            prediction = model.inference(tag_id, category_id, batch_items_feature)
            prediction = prediction.detach().cpu().numpy().tolist()
            one_tag_predictions += prediction
        one_tag_predictions = np.array(one_tag_predictions)
        result_matrix[:, i] = one_tag_predictions

    # metrics
    results = {'recall': np.zeros(len(topks)),
               'ndcg': np.zeros(len(topks))}
    max_k = max(topks)
    rating = torch.FloatTensor(result_matrix).cuda()
    # all the videos in test set do not exist in training set
    # no need to drop tags
    _, rating_K = torch.topk(rating, k=max_k)
    rating = rating.cpu().numpy()
    del rating
    ground_true = []
    for i in range(len(items)):
        ground_true.append(item_tags[items[i]])
    rating_list = [rating_K.cpu()]
    ground_true_list = [ground_true]
    X = zip(rating_list, ground_true_list)
    pre_results = []
    for x in X:
        pre_results.append(test_one_batch(x, topks))
    for result in pre_results:
        results['recall'] += result['recall']
        results['ndcg'] += result['ndcg']
    results['recall'] /= float(len(items))
    results['ndcg'] /= float(len(items))
    
    return results


def get_pair_tags(pair_list):
    item_tags = {}
    for item, tag in pair_list:
        if item not in item_tags:
            item_tags[item] = [tag]
        else:
            item_tags[item].append(tag)
    return item_tags


def Ranking(model, video_features_path, itemid_videohiddenid_map, valid_pair_list, train_pair_list, tag_num, topks, split_index):
    valid_item_tags = get_pair_tags(valid_pair_list)
    video_features = np.load(video_features_path, allow_pickle=True)[()]

    valid_result = get_metrics(model, valid_item_tags, video_features, itemid_videohiddenid_map, tag_num, topks, split_index)
    return valid_result


def test_Ranking(model, video_features_path, itemid_videohiddenid_map, test_pair_list, train_pair_list, tag_num, topks, split_index):
    test_item_tags = get_pair_tags(test_pair_list)
    video_features = np.load(video_features_path, allow_pickle=True)[()]

    test_result = get_metrics(model, test_item_tags, video_features, itemid_videohiddenid_map, tag_num, topks, split_index)

    return test_result



