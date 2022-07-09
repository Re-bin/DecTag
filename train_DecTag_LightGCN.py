import os
import time
import argparse
import numpy as np
import random

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn

import model
import data_util
import evaluate_GCN

start_time = time.time()


def shuffle(*arrays):
    shuffle_indices = np.arange(len(arrays[0]))
    np.random.shuffle(shuffle_indices)
    if len(arrays) == 1:
        result = arrays[0][shuffle_indices]
    else:
        result = tuple(x[shuffle_indices] for x in arrays)
    return result


def minibatch(*tensors, **kwargs):
    batch_size = kwargs.get('batch_size')
    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)


def BPR_train_one_epoch(dataset, model, optimizer, epoch, split_index, batch_size, weight_decay):
    model.set_model_type('train')
    model.train()
    items, pos_tags, neg_tags, confounders, category = data_util.uniform_sample_original_python(dataset)
    items = torch.Tensor(items).long().cuda()
    pos_tags = torch.Tensor(pos_tags).long().cuda()
    neg_tags = torch.Tensor(neg_tags).long().cuda()
    category = torch.Tensor(category).long().cuda()
    confounders = torch.FloatTensor(confounders).unsqueeze(dim=-1).cuda()
    items, pos_tags, neg_tags, category, confounders = shuffle(items, pos_tags, neg_tags, category, confounders)
    total_batch = len(items) // batch_size + 1
    aver_loss = 0

    if epoch <= 5:
        is_warmup = True
    else:
        is_warmup = False

    for (batch_i, (batch_items, batch_pos_tags, batch_neg_tags, batch_category, batch_confounders)) in enumerate(
            minibatch(items, pos_tags, neg_tags, category, confounders, batch_size=batch_size)):
        loss, reg_loss = model.bpr_loss(batch_items, batch_pos_tags, batch_neg_tags, batch_category, batch_confounders,
                                        is_warmup)
        optimizer.zero_grad()
        loss = loss + reg_loss * weight_decay
        loss.backward()
        optimizer.step()
        aver_loss += loss.cpu().item()
    aver_loss /= total_batch
    print("----" * 18)
    print("Split{} End Epoch {:03d} loss {:05f} ".format(split_index, epoch, aver_loss) + "costs " + time.strftime(
        "%H: %M: %S", time.gmtime(time.time() - start_time)))


random_seed = 1
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

######## ARGS ########
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

parser.add_argument("--warmup_lr",
                    type=float,
                    default=0.0001,
                    help="learning rate in burn-in phase")

parser.add_argument("--lr",
                    type=float,
                    default=0.001,
                    help="learning rate")

# For YT-8M-Causal(C.E. & I.T.)
# Change the number of epochs as 600
parser.add_argument("--epochs",
                    type=int,
                    default=400,
                    help="epochs")


args = parser.parse_args()
args.splits = args.splits.split(',')
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
print("splits:{}".format(args.splits))
print("Dataset:{}".format(args.dataset))
print("Device id:{}".format(args.gpu))
torch.backends.cudnn.deterministic = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

######## PREPARE DATASET ########
start_time = time.time()
item_features_path = 'data_split/{}/video_infos.npy'.format(args.dataset)
pair_list_path = 'data_split/{}/pair_list.npy'.format(args.dataset)
user_features_path = 'data_split/{}/user_features.npy'.format(args.dataset)
video_features_path = 'data_split/{}/video_features.npy'.format(args.dataset)

item_features = np.load(item_features_path, allow_pickle=True)[()]
pair_list = np.load(pair_list_path, allow_pickle=True)[()]
user_features = np.load(user_features_path, allow_pickle=True)[()]

for split_index in args.splits:
    data_set = args.dataset
    tag_num_path = 'data_split/{}/split_{}/tag_num.npy'.format(data_set, split_index)
    item_num_path = 'data_split/{}/split_{}/item_num.npy'.format(data_set, split_index)
    tag_num = np.load(tag_num_path, allow_pickle=True)[()]
    item_num = np.load(item_num_path, allow_pickle=True)[()]
    videohiddenid_itemid_map, itemid_videohiddenid_map, channelurl_userid_map, _, _ = data_util.generate_map_dict(
        item_features, user_features, pair_list)

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

    train_dataset = data_util.GCNData(tag_num, item_num, train_pair_list, test_pair_list, valid_pair_list,
                                      itemid_videohiddenid_map, video_features_path, item_confounder)

    ######## CREATE MODEL ########
    pre_train_path = None
    emodel = model.DecTag_LightGCN(tag_num=tag_num, item_num=item_num, confounders=confounders, num_layers=3,
                                    keep_prob=0.8, dropout=0.3,
                                    train_sparse_graph=train_dataset.get_sparse_graph(),
                                    item_feature_oemb=train_dataset.item_feature_oemb).to(device)
    if pre_train_path:
        emodel.load_state_dict(torch.load(pre_train_path))

    optimizer = optim.Adam(emodel.parameters(), lr=args.lr)

    ######## TRAINING ########
    epochs = args.epochs
    warmup_end = False
    best_state_dict = None
    best_valid_recall = -1
    best_valid_epoch = None

    for epoch in range(0, epochs + 1):

        # warm up
        if epoch <= 5:
            for params_group in optimizer.param_groups:
                params_group['lr'] = args.warmup_lr
        else:
            if warmup_end == False:
                warmup_end = True
                for params_group in optimizer.param_groups:
                    params_group['lr'] = args.lr

        BPR_train_one_epoch(train_dataset, emodel, optimizer, epoch, split_index, batch_size=4096, weight_decay=1e-4)

        #### EVALUATION ####
        if (epoch >= 350 and epoch % 10 == 0) or epoch == 0:
            result = evaluate_GCN.GCN_valid(train_dataset, emodel, epoch, start_time, split_index, valid_batch_size=1024, topks=[10])
            if result > best_valid_recall:
                best_valid_epoch = epoch
                best_valid_recall = result
                best_state_dict = emodel.state_dict()

    # SAVE MODEL
    torch.save(best_state_dict, 'check_point/{}/LightGCN/split_{}/{:03d}-{:.5f}.pth'.format(data_set, split_index, best_valid_epoch, best_valid_recall))
