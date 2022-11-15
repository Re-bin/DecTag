import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

class DecTag_NFM(nn.Module):
    def __init__(self, tag_num, confounders, num_layers, dropout, category_num=2, item_factor_num=1024, factor_num=64):
        super(DecTag_NFM, self).__init__()
        self.tag_num = tag_num
        self.confounders = confounders
        self.confounders_num = len(self.confounders)
        self.num_layers = num_layers
        self.dropout = dropout
        self.category_num = category_num
        self.factor_num = factor_num
        self.MLP_first_factor_num = factor_num * (2 ** (num_layers - 1))

        self.embed_item_1 = nn.Sequential(
            nn.Dropout(p=self.dropout),
            nn.Linear(item_factor_num, (item_factor_num + factor_num) // 2),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear((item_factor_num + factor_num) // 2, factor_num),
            nn.ReLU()
        )
        self.embed_tag_1 = nn.Embedding(tag_num, factor_num)
        self.embed_category_1 = nn.Embedding(category_num, factor_num)

        self.embed_item_2 = nn.Sequential(
            nn.Dropout(p=self.dropout),
            nn.Linear(item_factor_num, (item_factor_num + self.MLP_first_factor_num) // 2),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear((item_factor_num + self.MLP_first_factor_num) // 2, self.MLP_first_factor_num),
            nn.ReLU()
        )
        self.embed_tag_2 = nn.Embedding(tag_num, self.MLP_first_factor_num)
        self.embed_category_2 = nn.Embedding(category_num, self.MLP_first_factor_num)

        MLP_modules_tag_item = []
        for i in range(num_layers):
            input_size = factor_num * (2 ** (num_layers - i))
            MLP_modules_tag_item.append(nn.Dropout(p=self.dropout))
            MLP_modules_tag_item.append(nn.Linear(input_size, input_size // 2))
            MLP_modules_tag_item.append(nn.ReLU())
        self.MLP_layers_tag_item = nn.Sequential(*MLP_modules_tag_item)

        MLP_modules_tag_category = []
        for i in range(num_layers):
            input_size = factor_num * (2 ** (num_layers - i))
            MLP_modules_tag_category.append(nn.Dropout(p=self.dropout))
            MLP_modules_tag_category.append(nn.Linear(input_size, input_size // 2))
            MLP_modules_tag_category.append(nn.ReLU())
        self.MLP_layers_tag_category = nn.Sequential(*MLP_modules_tag_category)

        predict_size = factor_num * 2
        self.predict_layer_tag_item = nn.Linear(predict_size, 1)
        self.predict_layer_tag_category = nn.Linear(predict_size, 1)
        self.sigmoid = nn.Sigmoid()
        self._init_weight_()

    def _init_weight_(self):
        nn.init.normal_(self.embed_tag_1.weight, std=0.01)
        nn.init.normal_(self.embed_tag_2.weight, std=0.01)
        nn.init.normal_(self.embed_category_1.weight, std=0.01)
        nn.init.normal_(self.embed_category_2.weight, std=0.01)

        for m in self.MLP_layers_tag_item:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

        for m in self.MLP_layers_tag_category:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

        for m in self.embed_item_1:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

        for m in self.embed_item_2:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

        nn.init.kaiming_uniform_(self.predict_layer_tag_category.weight,
                                 a=1, nonlinearity='sigmoid')

        nn.init.kaiming_uniform_(self.predict_layer_tag_item.weight,
                                 a=1, nonlinearity='sigmoid')

        for m in self.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()

    def forward(self, tag, category, embed_item, confounder, is_warmup):

        if is_warmup:
            random_confounder = confounder
        else:
            batch_size = tag.shape[0]
            random_confounder = self.confounders[np.random.randint(0, self.confounders_num, batch_size)]
            random_confounder = torch.tensor(random_confounder, dtype=torch.float32).unsqueeze(dim=-1).cuda()

        # category embed for point-wise production
        embed_cate_1 = self.embed_category_1(category)
        embed_cate_1 = embed_cate_1 * random_confounder
        embed_cate_1 = torch.sum(embed_cate_1, 1)

        # tag embed for point-wise production
        embed_tag_1 = self.embed_tag_1(tag).squeeze()

        # item embed for point-wise production
        embed_item_1 = self.embed_item_1(embed_item)

        # output embed for point-wise production
        output_1_tag_item = embed_tag_1 * embed_item_1
        output_1_tag_category = embed_tag_1 * embed_cate_1

        # category embed for multi layer perception
        embed_cate_2 = self.embed_category_2(category)
        embed_cate_2 = embed_cate_2 * random_confounder
        embed_cate_2 = torch.sum(embed_cate_2, 1)

        # tag embed for multi layer perception
        embed_tag_2 = self.embed_tag_2(tag).squeeze()

        # item embed for multi layer perception
        embed_item_2 = self.embed_item_2(embed_item)

        # output embed
        interaction_tag_item = torch.cat((embed_tag_2, embed_item_2), -1)
        output_2_tag_item = self.MLP_layers_tag_item(interaction_tag_item)
        interaction_tag_category = torch.cat((embed_tag_2, embed_cate_2), -1)
        output_2_tag_category = self.MLP_layers_tag_category(interaction_tag_category)

        # predict value for tag_item
        concat_tag_item = torch.cat((output_1_tag_item, output_2_tag_item), -1)
        prediction_tag_item = self.predict_layer_tag_item(concat_tag_item)

        # predict value for tag_category
        concat_tag_category = torch.cat((output_1_tag_category, output_2_tag_category), -1)
        prediction_tag_category = self.predict_layer_tag_category(concat_tag_category)

        # final prediction
        prediction = prediction_tag_item * self.sigmoid(prediction_tag_category)
        prediction = self.sigmoid(prediction)

        return prediction.view(-1)

    def inference(self, tag, category, embed_item, sample_time=5):
        final_prediction = None
        for i in range(sample_time):
            batch_size = tag.shape[0]
            random_confounder = self.confounders[np.random.randint(0, self.confounders_num, batch_size)]
            random_confounder = torch.tensor(random_confounder, dtype=torch.float32).unsqueeze(dim=-1).cuda()

            # category embed for point-wise production
            embed_cate_1 = self.embed_category_1(category)
            embed_cate_1 = embed_cate_1 * random_confounder
            embed_cate_1 = torch.sum(embed_cate_1, 1)

            # tag embed for point-wise production
            embed_tag_1 = self.embed_tag_1(tag).squeeze()

            # item embed for point-wise production
            embed_item_1 = self.embed_item_1(embed_item)

            # output embed for point-wise production
            output_1_tag_item = embed_tag_1 * embed_item_1
            output_1_tag_category = embed_tag_1 * embed_cate_1

            # category embed for multi layer perception
            embed_cate_2 = self.embed_category_2(category)
            embed_cate_2 = embed_cate_2 * random_confounder
            embed_cate_2 = torch.sum(embed_cate_2, 1)

            # tag embed for multi layer perception
            embed_tag_2 = self.embed_tag_2(tag).squeeze()

            # item embed for multi layer perception
            embed_item_2 = self.embed_item_2(embed_item)

            # output embed for multi layer perception
            interaction_tag_item = torch.cat((embed_tag_2, embed_item_2), -1)
            output_2_tag_item = self.MLP_layers_tag_item(interaction_tag_item)
            interaction_tag_category = torch.cat((embed_tag_2, embed_cate_2), -1)
            output_2_tag_category = self.MLP_layers_tag_category(interaction_tag_category)

            # predict value for tag_item
            concat_tag_item = torch.cat((output_1_tag_item, output_2_tag_item), -1)
            prediction_tag_item = self.predict_layer_tag_item(concat_tag_item)

            # predict value for tag_category
            concat_tag_category = torch.cat((output_1_tag_category, output_2_tag_category), -1)
            prediction_tag_category = self.predict_layer_tag_category(concat_tag_category)

            # final prediction
            prediction = prediction_tag_item * self.sigmoid(prediction_tag_category)
            if i == 0:
                final_prediction = self.sigmoid(prediction)
            else:
                final_prediction += self.sigmoid(prediction)

        final_prediction /= sample_time
        return final_prediction.view(-1)


class DecTag_LightGCN(nn.Module):
    def __init__(self, tag_num, item_num, confounders, num_layers, keep_prob, dropout, train_sparse_graph,
                 item_feature_oemb, category_num=2, item_factor_num=1024, factor_num=64):
        super(DecTag_LightGCN, self).__init__()
        self.item_num = item_num
        self.tag_num = tag_num
        self.confounders = confounders
        self.confounders_num = len(self.confounders)
        self.num_layers = num_layers
        self.factor_num = factor_num
        self.dropout = dropout
        self.keep_prob = keep_prob
        self.item_feature_oemb = item_feature_oemb
        self.category_num = category_num
        self.factor_num = factor_num

        self.embedding_item = nn.Sequential(
            nn.Dropout(p=self.dropout),
            nn.Linear(item_factor_num, (item_factor_num + self.factor_num) // 2),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear((item_factor_num + self.factor_num) // 2, self.factor_num),
            nn.ReLU()
        )
        self.embedding_tag = nn.Embedding(self.tag_num, self.factor_num)
        self.embedding_category = nn.Embedding(self.category_num, self.factor_num)
        self.sigmoid = nn.Sigmoid()
        self.train_Graph = train_sparse_graph
        self.model_type = 'train'
        self.__init_weight()

    def __init_weight(self):
        nn.init.normal_(self.embedding_tag.weight, std=0.1)
        nn.init.normal_(self.embedding_category.weight, std=0.1)
        for m in self.embedding_item:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    def __dropout(self, keep_prob):
        if self.model_type == 'train':
            graph = self.__dropout_x(self.train_Graph, keep_prob)
        return graph

    def set_model_type(self, model_type):
        self.model_type = model_type

    def computer(self):
        items_emb = self.embedding_item(self.item_feature_oemb)
        tags_emb = self.embedding_tag.weight
        all_emb = torch.cat([items_emb, tags_emb])
        embs = [all_emb]

        if self.model_type == 'train':
            g_droped = self.__dropout(self.keep_prob)
            # convolution
            for layer in range(self.num_layers):
                all_emb = torch.sparse.mm(g_droped, all_emb)
                embs.append(all_emb)
            embs = torch.stack(embs, dim=1)
            light_out = torch.mean(embs, dim=1)
            items, tags = torch.split(light_out, [self.item_num, self.tag_num])
            return items, tags

        if self.model_type == 'test':
            g_droped = self.train_Graph
            # convolution
            for layer in range(self.num_layers):
                all_emb = torch.sparse.mm(g_droped, all_emb)
                embs.append(all_emb)
            embs = torch.stack(embs, dim=1)
            light_out = torch.mean(embs, dim=1)
            items, tags = torch.split(light_out, [self.item_num, self.tag_num])
            return items, tags

        if self.model_type == 'valid':
            g_droped = self.train_Graph
            # convolution
            for layer in range(self.num_layers):
                all_emb = torch.sparse.mm(g_droped, all_emb)
                embs.append(all_emb)
            embs = torch.stack(embs, dim=1)
            light_out = torch.mean(embs, dim=1)
            items, tags = torch.split(light_out, [self.item_num, self.tag_num])
            return items, tags

    def get_embedding(self, items, pos_tags, neg_tags):
        all_items, all_tags = self.computer()
        items_emb = all_items[items]
        pos_emb = all_tags[pos_tags]
        neg_emb = all_tags[neg_tags]
        pos_emb_ego = self.embedding_tag(pos_tags)
        neg_emb_ego = self.embedding_tag(neg_tags)
        return items_emb, pos_emb, neg_emb, pos_emb_ego, neg_emb_ego

    def bpr_loss(self, items, pos_tags, neg_tags, category, confounder, is_warmup):
        items_emb, pos_emb, neg_emb, pos_emb_ego, neg_emb_ego = self.get_embedding(items, pos_tags, neg_tags)
        reg_loss = (1 / 2) * (pos_emb_ego.norm(2).pow(2) + neg_emb_ego.norm(2).pow(2)) / float(len(items))

        if is_warmup:
            random_confounder = confounder
        else:
            batch_size = items.shape[0]
            random_confounder = self.confounders[np.random.randint(0, self.confounders_num, batch_size)]
            random_confounder = torch.tensor(random_confounder, dtype=torch.float32).unsqueeze(dim=-1).cuda()

        category_emb = self.embedding_category(category)
        category_emb = category_emb * random_confounder
        category_emb = torch.sum(category_emb, 1)

        # pos scores
        pos_scores = torch.mul(items_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        pos_con_scores = torch.mul(category_emb, pos_emb_ego)
        pos_con_scores = torch.sum(pos_con_scores, dim=1)
        pos_con_scores = self.sigmoid(pos_con_scores)
        pos_scores = pos_scores * pos_con_scores

        # neg scores
        neg_scores = torch.mul(items_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)
        neg_con_scores = torch.mul(category_emb, neg_emb_ego)
        neg_con_scores = torch.sum(neg_con_scores, dim=1)
        neg_con_scores = self.sigmoid(neg_con_scores)
        neg_scores = neg_scores * neg_con_scores

        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

        return loss, reg_loss

    def get_user_rating(self, items, sample_time=5):
        all_items, all_tags = self.computer()
        batch_item_num = items.shape[0]
        items_emb = all_items[items.long()]
        tags_emb = all_tags
        scores = torch.matmul(items_emb, tags_emb.t())
        rating = None
        for i in range(sample_time):
            # con_score
            category = torch.Tensor([0, 1]).long().cuda()
            category_emb = self.embedding_category(category)
            category_emb = category_emb.repeat(batch_item_num, 1, 1)
            random_confounder = self.confounders[np.random.randint(0, self.confounders_num, batch_item_num)]
            random_confounder = torch.tensor(random_confounder, dtype=torch.float32).unsqueeze(dim=-1).cuda()
            con_scores = category_emb * random_confounder
            con_scores = torch.sum(con_scores, dim=1)
            con_scores = con_scores.repeat(1, self.tag_num)
            con_score_shape = con_scores.shape
            con_scores = con_scores.view(con_score_shape[0], self.tag_num, con_score_shape[1] // self.tag_num)
            tags_emb_ego = self.embedding_tag.weight
            tags_emb_ego = tags_emb_ego.repeat(batch_item_num, 1, 1)
            con_scores = torch.mul(con_scores, tags_emb_ego)
            con_scores = torch.sum(con_scores, dim=2)
            con_scores = self.sigmoid(con_scores)
            if i == 0:
                rating = self.sigmoid(scores * con_scores)
            else:
                rating += self.sigmoid(scores * con_scores)
        rating /= sample_time
        return rating

    def forward(self, items, tags, category):
        all_items, all_tags = self.computer()
        items_emb = all_items[items]
        tags_emb = all_tags[items]
        tags_emb_ego = self.embedding_tag(tags)
        category_emb = self.embedding_category(category)
        category_emb = category_emb * self.confounder_prior
        category_emb = torch.sum(category_emb, 1)

        scores = torch.mul(items_emb, tags_emb)
        scores = torch.sum(scores, dim=1)
        con_scores = torch.mul(category_emb, tags_emb_ego)
        con_scores = torch.sum(con_scores, dim=1)
        con_scores = self.sigmoid(con_scores)
        scores = scores * con_scores

        return scores
