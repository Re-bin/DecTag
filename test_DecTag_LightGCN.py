import os
import time
import argparse
import numpy as np
import random
import torch
import pandas as pd
import torch.backends.cudnn as cudnn

import model
import data_util
import evaluate_GCN

start_time = time.time()

parser = argparse.ArgumentParser()
parser.add_argument("--dataset",
                    type=str,
                    help="Dataset")
parser.add_argument("--splits",
                    type=str,
                    default='1,2,3,4,5',
                    help="split index")
parser.add_argument("--gpu",
                    type=str,
                    default="0",
                    help="gpu card ID")
parser.add_argument("--seed",
                    type=int,
                    default=1,
                    help="random seed")
args = parser.parse_args()
args.splits = args.splits.split(',')
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random_seed = args.seed
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

Recall = []
NDCG = []
topks = [10, 20]
for k in topks:
    Recall.append([])
    NDCG.append([])
data_set = args.dataset

print("Start testing Dataset {} (splits {}) for DecTag-LightGCN".format(args.dataset, args.splits))


for i in range(len(args.splits)):
    ######## RESULT DAFAFRAME ########
    df_columns = ['split_index', 'model_state_dict_path']
    for k in topks:
        df_columns.append('recall@{}'.format(k))
    for k in topks:
        df_columns.append('ndcg@{}'.format(k))
    result_df = pd.DataFrame(columns=df_columns, index=[0])
    result_df['split_index'][0] = args.splits[i]

    ######## PREPARE DATASET ########
    split_index = args.splits[i]
    item_features_path = 'data_split/{}/video_infos.npy'.format(data_set)
    pair_list_path = 'data_split/{}/pair_list.npy'.format(data_set)
    user_features_path = 'data_split/{}/user_features.npy'.format(data_set)
    video_features_path = 'data_split/{}/video_features.npy'.format(data_set)
    tag_num_path = 'data_split/{}/split_{}/tag_num.npy'.format(data_set, split_index)
    item_num_path = 'data_split/{}/split_{}/item_num.npy'.format(data_set, split_index)

    item_features = np.load(item_features_path, allow_pickle=True)[()]
    pair_list = np.load(pair_list_path, allow_pickle=True)[()]
    user_features = np.load(user_features_path, allow_pickle=True)[()]
    tag_num = np.load(tag_num_path, allow_pickle=True)[()]
    item_num = np.load(item_num_path, allow_pickle=True)[()]
    videohiddenid_itemid_map, itemid_videohiddenid_map, channelurl_userid_map, _, _ = data_util.generate_map_dict(item_features, user_features, pair_list)

    train_pair_list_path = 'data_split/{}/split_{}/train_pair_list.npy'.format(data_set, split_index)
    valid_pair_list_path = 'data_split/{}/split_{}/valid_pair_list.npy'.format(data_set, split_index)
    test_pair_list_path = 'data_split/{}/split_{}/test_pair_list.npy'.format(data_set, split_index)
    item_confounder_path = 'data_split/{}/split_{}/item_confounder.npy'.format(data_set, split_index)
    confounders_path = 'data_split/{}/split_{}/confounders.npy'.format(data_set, split_index)

    train_pair_list = np.load(train_pair_list_path, allow_pickle=True)[()]
    valid_pair_list = np.load(valid_pair_list_path, allow_pickle=True)[()]
    test_pair_list = np.load(test_pair_list_path, allow_pickle=True)[()]
    item_confounder = np.load(item_confounder_path, allow_pickle=True)[()]
    confounders = np.load(confounders_path, allow_pickle=True)[()]
    dataset = data_util.GCNData(tag_num, item_num, train_pair_list, test_pair_list, valid_pair_list, itemid_videohiddenid_map, video_features_path, item_confounder)


    ######## CREATE MODEL ########
    pth_file = os.listdir("check_point/{}/LightGCN/split_{}".format(data_set, split_index))[0]
    pre_train_path = os.path.join("check_point/{}/LightGCN/split_{}".format(data_set, split_index), pth_file)
    result_df['model_state_dict_path'][0] = pre_train_path
    _model = model.DecTag_LightGCN(tag_num=tag_num, item_num=item_num, confounders=confounders, num_layers=3, keep_prob=0.8, dropout=0.3,
                       train_sparse_graph=dataset.get_sparse_graph(),
                       item_feature_oemb=dataset.item_feature_oemb).to(device)
    _model.load_state_dict(torch.load(pre_train_path))


    ######## TESTING ########
    _model.eval()
    results = evaluate_GCN.GCN_test(dataset, _model, split_index, test_batch_size=512, topks=topks)
    for j in range(len(topks)):
        result_df['recall@{}'.format(topks[j])][0] = results['recall'][j]
        result_df['ndcg@{}'.format(topks[j])][0] = results['ndcg'][j]

    ######## SAVING RESULT ########
    now_time = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())
    if os.exist('results') == False:
        os.mkdir('results')
    if os.exist('results/{}'.format(data_set)) == False:
        os.mkdir('results/{}'.format(data_set))
    if os.exist('results/{}/LightGCN/split_{}'.format(data_set, split_index)) == False:
        os.mkdir('results/{}/LightGCN/split_{}'.format(data_set, split_index))
    result_df.to_csv('results/{}/LightGCN/split_{}/test_result_{}.csv'.format(data_set, split_index, now_time), sep=',', header=True, index=False)

    for j in range(len(topks)):
        Recall[j].append(results['recall'][j])
        NDCG[j].append(results['ndcg'][j])

for j in range(len(topks)):
    print("Final(mean): Recall@{}: {:.5f} NDCG@{}: {:.5f}".format(topks[j], np.mean(np.array(Recall[j])), topks[j], np.mean(np.array(NDCG[j]))))
for j in range(len(topks)):
    print("Final(std): Recall@{}: {:.5f} NDCG@{}: {:.5f}".format(topks[j], np.std(np.array(Recall[j])), topks[j], np.std(np.array(NDCG[j]))))