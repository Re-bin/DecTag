import os
import time
import argparse
import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn

import model
import data_util
import evaluate_NFM

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

parser.add_argument("--seed",
                    type=int,
                    default=1,
                    help="random seed")

parser.add_argument("--lr",
                    type=float,
                    default=0.001,
                    help="learning rate")

parser.add_argument("--epochs",
                    type=int,
                    default=200,
                    help="epochs")

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


def train(data_set, split_index):
    ######## PREPARE DATASET ########
    start_time = time.time()
    item_features_path = 'data_split/{}/video_infos.npy'.format(data_set)
    pair_list_path = 'data_split/{}/pair_list.npy'.format(data_set)
    user_features_path = 'data_split/{}/user_features.npy'.format(data_set)
    video_features_path = 'data_split/{}/video_features.npy'.format(data_set)
    tag_num_path = 'data_split/{}/split_{}/tag_num.npy'.format(data_set, split_index)

    item_features = np.load(item_features_path, allow_pickle=True)[()]
    pair_list = np.load(pair_list_path, allow_pickle=True)[()]
    user_features = np.load(user_features_path, allow_pickle=True)[()]
    tag_num = np.load(tag_num_path, allow_pickle=True)[()]
    videohiddenid_itemid_map, itemid_videohiddenid_map, channelurl_userid_map, _, _ = data_util.generate_map_dict(
        item_features, user_features, pair_list)

    train_pair_list_path = 'data_split/{}/split_{}/train_pair_list.npy'.format(data_set, split_index)
    valid_pair_list_path = 'data_split/{}/split_{}/valid_pair_list.npy'.format(data_set, split_index)
    item_confounder_path = 'data_split/{}/split_{}/item_confounder.npy'.format(data_set, split_index)
    confounders_path = 'data_split/{}/split_{}/confounders.npy'.format(data_set, split_index)

    train_pair_list = np.load(train_pair_list_path, allow_pickle=True)[()]
    valid_pair_list = np.load(valid_pair_list_path, allow_pickle=True)[()]
    item_confounder = np.load(item_confounder_path, allow_pickle=True)[()]
    confounders = np.load(confounders_path, allow_pickle=True)[()]

    train_dataset = data_util.NFMData(tag_num, train_pair_list, itemid_videohiddenid_map, video_features_path, item_confounder)
    train_loader = data.DataLoader(train_dataset, batch_size=1024, shuffle=True, num_workers=5)

    ######## CREATE MODEL ########
    pre_train_path = None
    emodel = model.DecTag_NFM(tag_num=tag_num, confounders=confounders, num_layers=3, dropout=0.3, category_num=2).to(device)
    if pre_train_path:
        emodel.load_state_dict(torch.load(pre_train_path))
    optimizer = optim.Adam(emodel.parameters(), lr=args.lr)
    criterion = nn.BCELoss(reduction='sum')

    ######## TRAINING ########
    epochs = args.epochs
    warmup_end = False
    best_state_dict = None
    best_valid_recall = -1
    best_valid_epoch = None

    for epoch in range(0, epochs + 1):
        train_loss = 0.0
        step_cnt = 0.0
        train_loader.dataset.ng_sample()
        emodel.train()
        print("----" * 18)

        # warm up
        if epoch <= 5:
            is_warmup = True
            for params_group in optimizer.param_groups:
                params_group['lr'] = args.warmup_lr
        else:
            is_warmup = False
            if warmup_end == False:
                warmup_end = True
                for params_group in optimizer.param_groups:
                    params_group['lr'] = args.lr

        for tag_id, category_id, item_feature, label, confounder in train_loader:
            emodel.zero_grad()
            prediction = emodel(tag_id.cuda(), category_id.cuda(), item_feature.cuda(), confounder.cuda(), is_warmup)
            loss = criterion(prediction, label.cuda())
            loss += 0.1 * emodel.embed_tag_1.weight.norm()
            loss += 0.1 * emodel.embed_tag_2.weight.norm()
            loss += 0.1 * emodel.embed_category_1.weight.norm()
            loss += 0.1 * emodel.embed_category_1.weight.norm()
            train_loss += loss
            step_cnt += 1
            loss.backward()
            optimizer.step()
        train_loss /= step_cnt
        print("Split{} End Epoch {:03d} loss {:05f} ".format(split_index, epoch, train_loss) + "costs " + time.strftime("%H: %M: %S", time.gmtime(time.time() - start_time)))

        #### EVALUATION ####
        if (epoch >= 100 and epoch % 10 == 0) or epoch == 0:
            emodel.eval()
            topks = [10]
            valid_result = evaluate_NFM.Ranking(emodel, video_features_path, itemid_videohiddenid_map, valid_pair_list, train_pair_list, tag_num=tag_num, topks=topks, split_index=split_index)

            print('----' * 18)
            print("Runing Epoch {:03d} ".format(epoch) + "costs " + time.strftime("%H: %M: %S", time.gmtime(time.time() - start_time)))
            print("Split{} Validation Set: Recall@{}: {:.5f}".format(split_index, topks[0], valid_result['recall'][0]))

            if valid_result['recall'][0] > best_valid_recall:
                best_valid_epoch = epoch
                best_valid_recall = valid_result['recall'][0]
                best_state_dict = emodel.state_dict()

    # SAVE MODEL
    if os.exist('check_point') == False:
        os.mkdir('check_point')
    if os.exist('check_point/{}'.format(data_set)) == False:
        os.mkdir('check_point/{}'.format(data_set))
    if os.exist('check_point/{}/NFM/split_{}'.format(data_set, split_index)) == False:
        os.mkdir('check_point/{}/NFM/split_{}'.format(data_set, split_index))
    torch.save(best_state_dict, 'check_point/{}/NFM/split_{}/{:03d}-{:.5f}.pth'.format(data_set, split_index, best_valid_epoch, best_valid_recall))


if __name__ == "__main__":
    print("splits:{}".format(args.splits))
    print("Dataset:{}".format(args.dataset))
    print("Device id:{}".format(args.gpu))

    for i in args.splits:
        train(args.dataset, i)







