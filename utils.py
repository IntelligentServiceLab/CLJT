import heapq
import math

import torch
import random
import numpy as np
from torch_geometric.utils import structured_negative_sampling
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def sample_mini_batch(batch_size, edge_index):
    """Randomly samples indices of a minibatch given an adjacency matrix

    Args:
        batch_size (int): minibatch size
        edge_index (torch.Tensor): 2 by N list of edges

    Returns:
        tuple: user indices, positive item indices, negative item indices
    """

    edges = structured_negative_sampling(edge_index, num_nodes=edge_index[1].max().item() + 1)

    edges = torch.stack(edges, dim=0)

    indices = random.choices([i for i in range(edges[0].shape[0])], k=batch_size)
    batch = edges[:, indices]
    user_indices, pos_item_indices, neg_item_indices = batch[0], batch[1], batch[2]
    return user_indices, pos_item_indices, neg_item_indices




def bpr_loss(users_emb_final, users_emb_0, pos_items_emb_final, pos_items_emb_0, neg_items_emb_final, neg_items_emb_0,
             lambda_val):
    """Bayesian Personalized Ranking Loss as described in https://arxiv.org/abs/1205.2618

    Args:
        users_emb_final (torch.Tensor): e_u_k
        users_emb_0 (torch.Tensor): e_u_0
        pos_items_emb_final (torch.Tensor): positive e_i_k
        pos_items_emb_0 (torch.Tensor): positive e_i_0
        neg_items_emb_final (torch.Tensor): negative e_i_k
        neg_items_emb_0 (torch.Tensor): negative e_i_0
        lambda_val (float): lambda value for regularization loss term

    Returns:
        torch.Tensor: scalar bpr loss value
    """
    reg_loss = lambda_val * (users_emb_0.norm(2).pow(2) +
                             pos_items_emb_0.norm(2).pow(2) +
                             neg_items_emb_0.norm(2).pow(2))  # L2 loss

    pos_scores = torch.mul(users_emb_final, pos_items_emb_final)
    pos_scores = torch.sum(pos_scores, dim=-1)  # predicted scores of positive samples
    neg_scores = torch.mul(users_emb_final, neg_items_emb_final)
    neg_scores = torch.sum(neg_scores, dim=-1)  # predicted scores of negative samples

    loss = -torch.mean(torch.nn.functional.softplus(pos_scores - neg_scores)) + reg_loss

    return loss


def evaluation(model, edge_index, sparse_edge_index, exclude_edge_indices, k, lambda_val):
    """Evaluates model loss and metrics including recall, precision, ndcg @ k

    Args:
        model (LighGCN): lightgcn model
        edge_index (torch.Tensor): 2 by N list of edges for split to evaluate
        sparse_edge_index (sparseTensor): sparse adjacency matrix for split to evaluate
        exclude_edge_indices ([type]): 2 by N list of edges for split to discount from evaluation
        k (int): determines the top k items to compute metrics on
        lambda_val (float): determines lambda for bpr loss

    Returns:
        tuple: bpr loss, recall @ k, precision @ k, ndcg @ k
    """
    # get embeddings
    users_emb_final, users_emb_0, items_emb_final, items_emb_0 = model(sparse_edge_index)
    user_indices, pos_item_indices, neg_item_indices = structured_negative_sampling(edge_index,
                                                                                    contains_neg_self_loops=False,
                                                                                    num_nodes=edge_index[
                                                                                                  1].max().item() + 1)
    # print(f"{max(neg_item_indices)=}")

    users_emb_final, users_emb_0 = users_emb_final[user_indices], users_emb_0[user_indices]
    pos_items_emb_final, pos_items_emb_0 = items_emb_final[pos_item_indices], items_emb_0[pos_item_indices]
    neg_items_emb_final, neg_items_emb_0 = items_emb_final[neg_item_indices], items_emb_0[neg_item_indices]

    loss = bpr_loss(users_emb_final, users_emb_0, pos_items_emb_final, pos_items_emb_0, neg_items_emb_final,
                    neg_items_emb_0, lambda_val)

    recall, precision, ndcg, map_atk = get_metrics(model, edge_index, exclude_edge_indices, k)
    f1 = 2 * precision * recall / (precision + recall)

    return loss.item(), recall, precision, ndcg, map_atk, f1





def RecallPrecision_ATk(groundTruth, r, k):
    """Computers recall @ k and precision @ k

    Args:
        groundTruth (list): list of lists containing highly rated items of each user
        r (list): list of lists indicating whether each top k item recommended to each user
            is a top k ground truth item or not
        k (intg): determines the top k items to compute precision and recall on

    Returns:
        tuple: recall @ k, precision @ k
    """
    num_correct_pred = torch.sum(r, dim=-1)  # number of correctly predicted items per user
    # number of items liked by each user in the test set
    user_num_liked = torch.Tensor([len(groundTruth[i])
                                   for i in range(len(groundTruth))])
    recall = torch.mean(num_correct_pred / user_num_liked)
    precision = torch.mean(num_correct_pred) / k
    return recall.item(), precision.item()


def NDCGatK_r(groundTruth, r, k):
    """Computes Normalized Discounted Cumulative Gain (NDCG) @ k

    Args:
        groundTruth (list): list of lists containing highly rated items of each user
        r (list): list of lists indicating whether each top k item recommended to each user
            is a top k ground truth item or not
        k (int): determines the top k items to compute ndcg on

    Returns:
        float: ndcg @ k
    """
    assert len(r) == len(groundTruth)

    test_matrix = torch.zeros((len(r), k))

    for i, items in enumerate(groundTruth):
        length = min(len(items), k)
        test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = torch.sum(max_r * 1. / torch.log2(torch.arange(2, k + 2)), axis=1)
    dcg = r * (1. / torch.log2(torch.arange(2, k + 2)))
    dcg = torch.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg / idcg
    ndcg[torch.isnan(ndcg)] = 0.
    return torch.mean(ndcg).item()


def get_metrics(model, edge_index, exclude_edge_indices, k):
    """Computes the evaluation metrics: recall, precision, and ndcg @ k

    Args:
        model (LighGCN): lightgcn model
        edge_index (torch.Tensor): 2 by N list of edges for split to evaluate
        exclude_edge_indices ([type]): 2 by N list of edges for split to discount from evaluation
        k (int): determines the top k items to compute metrics on

    Returns:
        tuple: recall @ k, precision @ k, ndcg @ k
    """
    user_embedding = model.users_emb.weight
    item_embedding = model.items_emb.weight

    # get ratings between every user and item - shape is num users x num movies
    rating = torch.matmul(user_embedding, item_embedding.T)

    for exclude_edge_index in exclude_edge_indices:
        # gets all the positive items for each user from the edge index
        user_pos_items = get_user_positive_items(exclude_edge_index)
        # get coordinates of all edges to exclude
        exclude_users = []
        exclude_items = []
        for user, items in user_pos_items.items():
            exclude_users.extend([user] * len(items))
            exclude_items.extend(items)

        # set ratings of excluded edges to large negative value
        rating[exclude_users, exclude_items] = -(1 << 10)

    # get the top k recommended items for each user
    _, top_K_items = torch.topk(rating, k=k)

    # get all unique users in evaluated split
    users = edge_index[0].unique()

    test_user_pos_items = get_user_positive_items(edge_index)

    # convert test user pos items dictionary into a list
    test_user_pos_items_list = [
        test_user_pos_items[user.item()] for user in users]

    # determine the correctness of topk predictions
    r = []
    for user in users:
        ground_truth_items = test_user_pos_items[user.item()]
        label = list(map(lambda x: x in ground_truth_items, top_K_items[user]))
        r.append(label)
    r = torch.Tensor(np.array(r).astype('float'))

    recall, precision = RecallPrecision_ATk(test_user_pos_items_list, r, k)
    ndcg = NDCGatK_r(test_user_pos_items_list, r, k)
    map_at_k = MAP(test_user_pos_items_list, r)

    return recall, precision, ndcg, map_at_k


def my_get_metrics(model, edge_index, exclude_edge_indices, k, mashup_pooled_output, api_pooled_output):
    """Computes the evaluation metrics: recall, precision, and ndcg @ k

    Args:
        model (LighGCN): lightgcn model
        edge_index (torch.Tensor): 2 by N list of edges for split to evaluate
        exclude_edge_indices ([type]): 2 by N list of edges for split to discount from evaluation
        k (int): determines the top k items to compute metrics on

    Returns:
        tuple: recall @ k, precision @ k, ndcg @ k
    """
    user_embedding = model.users_emb.weight
    item_embedding = model.items_emb.weight
    m = mashup_pooled_output + user_embedding
    a = item_embedding + api_pooled_output

    # get ratings between every user and item - shape is num users x num movies
    # rating = torch.matmul(user_embedding, item_embedding.T)
    rating = torch.matmul(m, a.T)
    for exclude_edge_index in exclude_edge_indices:
        # gets all the positive items for each user from the edge index
        user_pos_items = get_user_positive_items(exclude_edge_index)
        # get coordinates of all edges to exclude
        exclude_users = []
        exclude_items = []
        for user, items in user_pos_items.items():
            exclude_users.extend([user] * len(items))
            exclude_items.extend(items)

        # set ratings of excluded edges to large negative value
        rating[exclude_users, exclude_items] = -(1 << 10)

    # get the top k recommended items for each user
    _, top_K_items = torch.topk(rating, k=k)

    # get all unique users in evaluated split
    users = edge_index[0].unique()

    test_user_pos_items = get_user_positive_items(edge_index)

    # convert test user pos items dictionary into a list
    test_user_pos_items_list = [
        test_user_pos_items[user.item()] for user in users]

    # determine the correctness of topk predictions
    r = []
    for user in users:
        ground_truth_items = test_user_pos_items[user.item()]
        label = list(map(lambda x: x in ground_truth_items, top_K_items[user]))
        r.append(label)
    r = torch.Tensor(np.array(r).astype('float'))

    recall, precision = RecallPrecision_ATk(test_user_pos_items_list, r, k)
    ndcg = NDCGatK_r(test_user_pos_items_list, r, k)
    map_at_k = MAP(test_user_pos_items_list, r)

    return recall, precision, ndcg, map_at_k




def MAP(test_user_pos_items_list, r):
    """
    Computes the Mean Average Precision (MAP)

    Args:
        test_user_pos_items_list (list): List of positive items for each user in the test set
        r (torch.Tensor): Tensor indicating the correctness of top-k predictions for each user

    Returns:
        float: Mean Average Precision (MAP)
    """
    APs = []
    for ground_truth_items, labels in zip(test_user_pos_items_list, r):
        if sum(labels) == 0:
            APs.append(0)
        else:
            num_correct = 0
            precisions = []
            for i, label in enumerate(labels):
                if label == 1:
                    num_correct += 1
                    precisions.append(num_correct / (i + 1))
            AP = sum(precisions) / len(ground_truth_items)
            APs.append(AP)
    MAP = np.mean(APs)
    return MAP


def get_user_positive_items(edge_index):
    """Generates dictionary of positive items for each user

    Args:
        edge_index (torch.Tensor): 2 by N list of edges

    Returns:
        dict: dictionary of positive items for each user
    """
    user_pos_items = {}
    for i in range(edge_index.shape[1]):
        user = edge_index[0][i].item()
        item = edge_index[1][i].item()
        if user not in user_pos_items:
            user_pos_items[user] = []
        user_pos_items[user].append(item)
    return user_pos_items


def CL_loss(view1, view2, temperature: float, b_cos: bool = True):
    """
    Args:
        view1: (torch.Tensor - N x D)
        view2: (torch.Tensor - N x D)
        temperature: float
        b_cos (bool)

    Return: Average InfoNCE Loss
    """
    if b_cos:
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)

    pos_score = (view1 @ view2.T) / temperature
    score = torch.diag(F.log_softmax(pos_score, dim=1))
    return -score.mean()


class ModelConfig:
    def __init__(self):
        self.embedding_dim = 256
        self.snafm_emb = 256
        self.num_layers = 3
        self.n_epoch = 10000
        self.train_batch_size = 64
        self.test_batch_size = 128
        self.lr = 1e-3
        self.weight_decay = 0
        self.lamda = 1e-5
        self.K = 3
        self.input_dim = 768
        self.hidden_dim = 256
        self.temperature = 0.02
        self.cl_rate = 0.015
        self.lambda_2 = 0.01
        self.theta = 0.05
        self.lambda_1 = 0.02
        self.reg = 0.0001
        self.N = 3
        self.eval_steps = 1000


def l2_reg_loss(reg, *args):
    emb_loss = 0
    for emb in args:
        emb_loss += torch.norm(emb, p=2) / emb.shape[0]
    return emb_loss * reg


class Metric(object):
    def __init__(self):
        pass

    @staticmethod
    def hits(origin, res):
        hit_count = {}
        for user in origin:
            items = list(origin[user].keys())
            predicted = [item[0] for item in res[user]]
            hit_count[user] = len(set(items).intersection(set(predicted)))
        return hit_count

    @staticmethod
    def hit_ratio(origin, hits):
        """
        Note: This type of hit ratio calculates the fraction:
         (# retrieved interactions in the test set / #all the interactions in the test set)
        """
        total_num = 0
        for user in origin:
            items = list(origin[user].keys())
            total_num += len(items)
        hit_num = 0
        for user in hits:
            hit_num += hits[user]
        return round(hit_num / total_num, 5)

    @staticmethod
    def my_hit_ratio(hits, N):
        """
        Note: This type of hit ratio calculates the fraction:
         (# retrieved interactions in the test set / #all the interactions in the test set)
        """
        return round(hits / N, 5)

    @staticmethod
    def precision(hits, N):
        prec = sum([hits[user] for user in hits])
        return round(prec / (len(hits) * N), 5)

    @staticmethod
    def recall(hits, origin):
        recall_list = [hits[user] / len(origin[user]) for user in hits]
        recall = round(sum(recall_list) / len(recall_list), 5)
        return recall

    @staticmethod
    def my_precision(hits_num, N):
        prec = sum(hits_num.values())  # Summing up all the hit counts
        return round(prec / (len(hits_num) * N), 5)

    @staticmethod
    def my_recall(hits, origin):
        recall_list = [hits[user] / len(origin[user]) for user in hits.keys()]
        recall = round(sum(recall_list) / len(recall_list), 5)
        return recall

    @staticmethod
    def F1(prec, recall):
        if (prec + recall) != 0:
            return round(2 * prec * recall / (prec + recall), 5)
        else:
            return 0

    @staticmethod
    def MAE(res):
        error = 0
        count = 0
        for entry in res:
            error += abs(entry[2] - entry[3])
            count += 1
        if count == 0:
            return error
        return round(error / count, 5)

    @staticmethod
    def RMSE(res):
        error = 0
        count = 0
        for entry in res:
            error += (entry[2] - entry[3]) ** 2
            count += 1
        if count == 0:
            return error
        return round(math.sqrt(error / count), 5)

    @staticmethod
    def NDCG(origin, res, N):
        sum_NDCG = 0
        for user in res:
            DCG = 0
            IDCG = 0
            # 1 = related, 0 = unrelated
            for n, item in enumerate(res[user]):
                if item[0] in origin[user]:
                    DCG += 1.0 / math.log(n + 2, 2)
            for n, item in enumerate(list(origin[user].keys())[:N]):
                IDCG += 1.0 / math.log(n + 2, 2)
            sum_NDCG += DCG / IDCG
        return round(sum_NDCG / len(res), 5)


class My_metric(object):
    def __init__(self):
        pass

    @staticmethod
    # def calculate_precision(actual, recommended,N):
    #     k=(set(actual) & set(recommended))
    #     Numerator=len(k)
    #     return Numerator/N
    def calculate_precision(actual, recommended):
        # 计算实际值和推荐结果的交集数量
        intersection_count = 0
        for item in recommended:
            if item in actual:
                intersection_count += 1

        # 分母为推荐结果的数量
        denominator = len(recommended)

        # 计算精确度
        precision = intersection_count / denominator if denominator != 0 else 0

        return precision

    @staticmethod
    # def calculate_recall(actual, recommended):
    #     denominator =len(actual)
    #     Numerator= len(set(actual) & set(recommended))
    #     return Numerator/denominator
    def calculate_recall(actual, recommended):
        # 计算实际值的数量
        denominator = len(actual)

        # 计算实际值和推荐结果的交集数量
        intersection_count = 0
        for item in recommended:
            if item in actual:
                intersection_count += 1

        # 计算召回率
        recall = intersection_count / denominator if denominator != 0 else 0

        return recall

    @staticmethod
    def calculate_f1_score(precision, recall):
        return 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    @staticmethod
    def calculate_MAP(actual, recommended, N):
        # Initialize variables
        num_relevant_services = len(actual)
        num_retrieved_relevant = 0
        precision_at_n = 0.0
        # Loop through the recommended services up to position N
        for j in range(N):
            # Check if the recommended service at position j is relevant
            if recommended[j] in actual:
                # Increment the count of retrieved relevant services
                num_retrieved_relevant += 1
                # Calculate precision at position j
                precision_at_n += num_retrieved_relevant / (j + 1)

        # Calculate average precision for the current application
        if num_relevant_services > 0:
            average_precision = precision_at_n / min(num_relevant_services, N)

        return average_precision

    @staticmethod
    def NDCG(actual_interests, recommended_items, N):
        DCG = 0
        IDCG = 0

        # 计算DCG
        for n, item in enumerate(recommended_items):
            relevance = 1 if item in actual_interests else 0
            DCG += relevance / math.log(n + 2, 2)

        # 计算IDCG
        for n in range(min(N, len(actual_interests))):
            IDCG += 1 / math.log(n + 2, 2)

        # 计算NDCG
        if IDCG == 0:
            return 0
        else:
            return round(DCG / IDCG, 5)


def normalize_sample_label(mashup_emb_temp, api_emb_temp, neg_api_emb_temp):
    # 计算均值和标准差
    # mashup_mean = torch.mean(mashup_emb_temp)
    # mashup_std = torch.std(mashup_emb_temp)
    #
    # api_mean = torch.mean(api_emb_temp)
    # api_std = torch.std(api_emb_temp)
    #
    # neg_api_mean = torch.mean(neg_api_emb_temp)
    # neg_api_std = torch.std(neg_api_emb_temp)
    #
    # # Z-Score 归一化
    # mashup_emb = (mashup_emb_temp - mashup_mean) / mashup_std
    # api_emb = (api_emb_temp - api_mean) / api_std
    # neg_api_emb = (neg_api_emb_temp - neg_api_mean) / neg_api_std
    mashup_emb = mashup_emb_temp
    api_emb = api_emb_temp
    neg_api_emb = neg_api_emb_temp

    positive_samples = torch.cat((mashup_emb, api_emb), dim=1)
    negative_samples = torch.cat((mashup_emb, neg_api_emb), dim=1)
    num_positive_samples = positive_samples.shape[0]
    num_negative_samples = negative_samples.shape[0]

    # 创建对应的标签数组
    positive_labels = np.ones((num_positive_samples,))
    negative_labels = np.zeros((num_negative_samples,))
    positive_labels_tensor = torch.tensor(positive_labels, dtype=torch.float32)
    negative_labels_tensor = torch.tensor(negative_labels, dtype=torch.float32)
    samples = torch.cat([positive_samples, negative_samples], dim=0)
    labels = torch.cat([positive_labels_tensor, negative_labels_tensor], dim=0)
    samples.to(device)
    labels.to(device)

    return samples, labels
