import random
import numpy as np
import torch
import torch.utils.data as data
from scipy.sparse import csr_matrix

class NFMData(data.Dataset):
    def __init__(self, tag_num, item_tag_pair, itemid_videohiddenid_map, video_features_path, item_confounder):
        self.tag_num = tag_num
        self.pos_item_tag_pair = item_tag_pair
        self.itemid_videohiddenid_map = itemid_videohiddenid_map
        self.video_features_path = video_features_path
        self.video_features = np.load(video_features_path, allow_pickle=True)[()]
        self.item_confounder = item_confounder
        self.pos_labels = []
        self.item_tags_dict = {}

        for item, tag in item_tag_pair:
            if item not in self.item_tags_dict:
                self.item_tags_dict[item] = [tag]
            else:
                self.item_tags_dict[item].append(tag)
            self.pos_labels.append(np.float32(1.0))

        self.item_tag_pair = self.pos_item_tag_pair
        self.labels = self.pos_labels

    def ng_sample(self):
        neg_item_tag_pair = []
        neg_labels = []
        for item in self.item_tags_dict:
            tags = self.item_tags_dict[item]
            ng_tag = random.randint(0, self.tag_num - 1)
            while ng_tag in tags:
                ng_tag = random.randint(0, self.tag_num - 1)
            neg_item_tag_pair.append([item, ng_tag])
            neg_labels.append(np.float32(0.0))
        neg_item_tag_pair = np.array(neg_item_tag_pair)
        self.item_tag_pair = np.row_stack((self.pos_item_tag_pair, neg_item_tag_pair))
        self.labels = self.pos_labels + neg_labels

    def __getitem__(self, index):
        item_id, tag_id = self.item_tag_pair[index]
        item_feature = np.array(self.video_features[self.itemid_videohiddenid_map[item_id]])
        tag_id = torch.LongTensor([tag_id])
        category_id = torch.LongTensor([0, 1])
        item_feature = torch.FloatTensor(item_feature)
        label = self.labels[index]
        confounder = torch.tensor(self.item_confounder[item_id], dtype=torch.float32).unsqueeze(dim=-1)

        return tag_id, category_id, item_feature, label, confounder

    def __len__(self):
        return len(self.item_tag_pair)

class GCNData(data.Dataset):
    def __init__(self, tag_num, item_num, train_item_tag_pair, test_item_tag_pair, valid_item_tag_pair, itemid_videohiddenid_map, video_features_path, item_confounder):
        self.tag_num = tag_num
        self.item_num = item_num
        self.train_pos_item_tag_pair = train_item_tag_pair
        self.test_pos_item_tag_pair = test_item_tag_pair
        self.valid_pos_item_tag_pair = valid_item_tag_pair
        self.itemid_videohiddenid_map = itemid_videohiddenid_map
        self.video_features_path = video_features_path
        self.video_features = np.load(video_features_path, allow_pickle=True)[()]
        self.item_confounder = item_confounder
        self.pos_labels = []
        self.item_tags_dict = {}

        item_feature_oemb = []
        for i in range(item_num):
            item_oembed = self.video_features[self.itemid_videohiddenid_map[i]]
            item_feature_oemb.append(item_oembed)
        self._item_feature_oemb = torch.FloatTensor(item_feature_oemb).cuda()


        self.train_item = self.train_pos_item_tag_pair[:, 0]
        self.train_unique_item = np.unique(self.train_item)
        self.train_tag = self.train_pos_item_tag_pair[:, 1]
        self.train_Graph = None

        self.test_item = self.test_pos_item_tag_pair[:, 0]
        self.test_unique_item = np.unique(self.test_item)
        self.test_tag = self.test_pos_item_tag_pair[:, 1]
        self.train_test_pos_item_tag_pair = np.concatenate((self.train_pos_item_tag_pair, self.test_pos_item_tag_pair), axis=0)
        self.train_test_item = self.train_test_pos_item_tag_pair[:, 0]
        self.train_test_unique_item = np.unique(self.train_test_item)
        self.train_test_tag = self.train_test_pos_item_tag_pair[:, 1]

        self.valid_item = self.valid_pos_item_tag_pair[:, 0]
        self.valid_unique_item = np.unique(self.valid_item)
        self.valid_tag = self.valid_pos_item_tag_pair[:, 1]
        self.train_valid_pos_item_tag_pair = np.concatenate((self.train_pos_item_tag_pair, self.valid_pos_item_tag_pair), axis=0)
        self.train_valid_item = self.train_valid_pos_item_tag_pair[:, 0]
        self.train_valid_unique_item = np.unique(self.train_valid_item)
        self.train_valid_tag = self.train_valid_pos_item_tag_pair[:, 1]

        self.train_item_tag_net = csr_matrix((np.ones(len(self.train_item)), (self.train_item, self.train_tag)), shape=(self.item_num, self.tag_num))
        self.test_item_tag_net = csr_matrix((np.ones(len(self.test_item)), (self.test_item, self.test_tag)), shape=(self.item_num, self.tag_num))
        self.valid_item_tag_net = csr_matrix((np.ones(len(self.valid_item)), (self.valid_item, self.valid_tag)), shape=(self.item_num, self.tag_num))

        self._train_all_pos = self.get_train_item_pos_tags(list(range(self.item_num)))
        self.train_all_neg = []
        all_tags = set(range(self.tag_num))
        for i in range(self.item_num):
            pos = set(self._train_all_pos[i])
            neg = all_tags - pos
            self.train_all_neg.append(np.array(list(neg)))

        self.__test_dict = self.__build_test()

        self.__valid_dict = self.__build_valid()

    @property
    def n_items(self):
        return self.item_num

    @property
    def n_tags(self):
        return self.tag_num

    @property
    def train_data_size(self):
        return len(self.train_item)

    @property
    def train_all_pos(self):
        return self._train_all_pos

    @property
    def item_feature_oemb(self):
        return self._item_feature_oemb

    @property
    def test_dict(self):
        return self.__test_dict

    @property
    def valid_dict(self):
        return self.__valid_dict

    def get_train_item_pos_tags(self, items):
        pos_items = []
        for item in items:
            pos_items.append(self.train_item_tag_net[item].nonzero()[1])
        return pos_items

    def get_train_item_neg_tags(self, items):
        neg_items = []
        for item in items:
            neg_items.append(self.train_all_neg[item])
        return neg_items

    def get_item_confounder(self, item):
        return self.item_confounder[item]

    def get_sparse_graph(self):
        if self.train_Graph is None:
            item_dim = torch.LongTensor(self.train_item)
            tag_dim = torch.LongTensor(self.train_tag)
            first_sub = torch.stack([item_dim, tag_dim + self.item_num])
            second_sub = torch.stack([tag_dim + self.item_num, item_dim])
            index = torch.cat([first_sub, second_sub], dim=1)
            data = torch.ones(index.size(-1)).int()
            self.train_Graph = torch.sparse.IntTensor(index, data, torch.Size([self.item_num + self.tag_num, self.item_num + self.tag_num]))
            dense = self.train_Graph.to_dense()
            D = torch.sum(dense, dim=1).float()
            D[D == 0.] = 1.
            D_sqrt = torch.sqrt(D).unsqueeze(dim=0)
            dense = dense / D_sqrt
            dense = dense / D_sqrt.t()
            index = dense.nonzero()
            data = dense[dense >= 1e-9]
            assert len(index) == len(data)
            self.train_Graph = torch.sparse.FloatTensor(index.t(), data, torch.Size([self.item_num + self.tag_num, self.item_num + self.tag_num]))
            self.train_Graph = self.train_Graph.coalesce().cuda()
        return self.train_Graph

    def __build_test(self):
        test_data = {}
        for i, tag in enumerate(self.test_tag):
            item = self.test_item[i]
            if test_data.get(item):
                test_data[item].append(tag)
            else:
                test_data[item] = [tag]
        return test_data

    def __build_valid(self):
        valid_data = {}
        for i, tag in enumerate(self.valid_tag):
            item = self.valid_item[i]
            if valid_data.get(item):
                valid_data[item].append(tag)
            else:
                valid_data[item] = [tag]
        return valid_data


    def __len__(self):
        return len(self.train_unique_item)

def uniform_sample_original_python(dataset):
    dataset: GCNData
    item_num = dataset.train_data_size
    all_pos = dataset.train_all_pos
    items = []
    pos_tags = []
    neg_tags = []
    confounders = []
    category = []
    while(item_num >= 0):
        item = np.random.randint(0, dataset.n_items)
        pos_for_item = all_pos[item]
        if len(pos_for_item) == 0:
            continue
        pos_index = np.random.randint(0, len(pos_for_item))
        pos_tag = pos_for_item[pos_index]
        pos_tags.append(pos_tag)
        while True:
            neg_tag = np.random.randint(0, dataset.n_tags)
            if neg_tag in pos_for_item:
                continue
            else:
                break
        items.append(item)
        neg_tags.append(neg_tag)
        confounders.append(dataset.get_item_confounder(item))
        category.append([0, 1])
        item_num -= 1

    return items, pos_tags, neg_tags, confounders, category

def generate_map_dict(item_features, user_features, pair_list):
    videohiddenid_itemid_map = {}
    itemid_videohiddenid_map = {}
    channelurl_userid_map = {}
    tagname_tagid_map = {}
    item_user_dict = {}

    itemid = 0
    userid = 0
    tagid = 0

    for videohiddenid in item_features:
        tags, _ = item_features[videohiddenid]
        if videohiddenid not in videohiddenid_itemid_map:
            videohiddenid_itemid_map[videohiddenid] = itemid
            itemid_videohiddenid_map[itemid] = videohiddenid
            itemid += 1
        for tag in tags:
            if tag not in tagname_tagid_map:
                tagname_tagid_map[tag] = tagid
                tagid += 1

    for channelurl in user_features:
        if channelurl not in channelurl_userid_map:
            channelurl_userid_map[channelurl] = userid
            userid += 1

    for user, videohiddenid in pair_list:
        itemid = videohiddenid_itemid_map[videohiddenid]
        item_user_dict[itemid] = user

    return videohiddenid_itemid_map, itemid_videohiddenid_map, channelurl_userid_map, tagname_tagid_map, item_user_dict



