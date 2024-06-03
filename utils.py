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


def my_sample_mini_batch(batch_size, edge_index, mashup_emb, api_emb):
    edges = structured_negative_sampling(edge_index, num_nodes=edge_index[1].max().item() + 1)

    edges = torch.stack(edges, dim=0)

    indices = random.choices([i for i in range(edges[0].shape[0])], k=batch_size)
    batch = edges[:, indices]
    user_indices, pos_item_indices, neg_item_indices = batch[0], batch[1], batch[2]
    mashup_input, api_input, neg_api_input = mashup_emb[user_indices], api_emb[pos_item_indices], api_emb[
        neg_item_indices]
    return user_indices, pos_item_indices, neg_item_indices, mashup_input, api_input, neg_api_input


def my_sample_mini_batch_test(batch_size, edge_index):
    edges = structured_negative_sampling(edge_index, num_nodes=edge_index[1].max().item() + 1)

    edges = torch.stack(edges, dim=0)

    indices = random.choices([i for i in range(edges[0].shape[0])], k=batch_size)
    batch = edges[:, indices]
    user_indices, pos_item_indices, neg_item_indices = batch[0], batch[1], batch[2]

    return user_indices


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


def my_evaluation_1(model, edge_index, sparse_edge_index, exclude_edge_indices, k, lambda_val, mashup_emb, api_emb,
                    temp):
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

    user_indices, pos_item_indices, neg_item_indices = structured_negative_sampling(edge_index,
                                                                                    contains_neg_self_loops=False,
                                                                                    num_nodes=edge_index[
                                                                                                  1].max().item() + 1)

    users_emb_final, users_emb_0, items_emb_final, items_emb_0, api_pooled_output, mashup_pooled_output = model(
        sparse_edge_index, mashup_emb, api_emb)

    # print(f"{max(neg_item_indices)=}")

    users_emb_final, users_emb_0 = users_emb_final[user_indices], users_emb_0[user_indices]
    pos_items_emb_final, pos_items_emb_0 = items_emb_final[pos_item_indices], items_emb_0[pos_item_indices]
    neg_items_emb_final, neg_items_emb_0 = items_emb_final[neg_item_indices], items_emb_0[neg_item_indices]

    loss = bpr_loss(users_emb_final, users_emb_0, pos_items_emb_final, pos_items_emb_0, neg_items_emb_final,
                    neg_items_emb_0, lambda_val)
    cl_loss_m = CL_loss(mashup_pooled_output, users_emb_final, temperature=temp)
    cl_loss_a = CL_loss(api_pooled_output, items_emb_final, temperature=temp)

    recall, precision, ndcg, map_atk = my_get_metrics(model, edge_index, exclude_edge_indices, k, mashup_pooled_output,
                                                      api_pooled_output)
    f1 = 2 * precision * recall / (precision + recall)

    return loss.item(), recall, precision, ndcg, map_atk, f1, cl_loss_a.item(), cl_loss_m.item()


def my_evaluation_2(model, edge_index, train_index, test_index, k, mashup, api, batch_size, label):
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

    # my_train_set = set(edge_index[0, train_index].tolist())
    # my_test_set = set(edge_index[0, test_index].tolist())
    # test_index_list = list(my_test_set - my_train_set)

    my_test_set = edge_index[0, test_index].tolist()

    precision_average, recall_average, f1_score_average, MAP_average, ndcg_average = my_get_metrics_2(model,
                                                                                                      my_test_set, k,
                                                                                                      batch_size,
                                                                                                      mashup, api,
                                                                                                      label)

    return precision_average, recall_average, f1_score_average, MAP_average, ndcg_average


def my_evaluation_3(model, edge_index, test_edge_index, test_sparse_edge_index, k, lambda_val, mashup_emb, api_emb,
                    temp, sanfm_model, batch_size, label):
    """Evaluates model loss and metrics including recall, precision, ndcg @ k

    Args:
        model (LighGCN): lightgcn model
        edge_index (torch.Tensor): 2 by N list of edges for split to evaluate
        sparse_edge_index (sparseTensor): sparse adjacency matrix for split to evaluate
        exclude_edge_indices ([type]): 2 by N list of edges for split to discount from evaluationolll
        k (int): determines the top k items to compute metrics on
        lambda_val (float): determines lambda for bpr loss

    Returns:
        tuple: bpr loss, recall @ k, precision @ k, ndcg @ k
    """

    # get embeddings

    user_indices, pos_item_indices, neg_item_indices = structured_negative_sampling(test_edge_index,
                                                                                    contains_neg_self_loops=False,
                                                                                    num_nodes=test_edge_index[
                                                                                                  1].max().item() + 1)

    users_emb_final, users_emb_0, items_emb_final, items_emb_0, api_pooled_output, mashup_pooled_output = model(
        test_sparse_edge_index, mashup_emb, api_emb)

    # print(f"{max(neg_item_indices)=}")

    users_emb_final1, users_emb_0 = users_emb_final[user_indices], users_emb_0[user_indices]
    pos_items_emb_final, pos_items_emb_0 = items_emb_final[pos_item_indices], items_emb_0[pos_item_indices]
    neg_items_emb_final, neg_items_emb_0 = items_emb_final[neg_item_indices], items_emb_0[neg_item_indices]
    mashup_cl_emb = mashup_pooled_output[user_indices]
    api_cl_emb = api_pooled_output[pos_item_indices]

    my_test_set = edge_index[0, user_indices].tolist()

    test_label = [label[j] for j in my_test_set]

    mashup_perc_emb = mashup_pooled_output[user_indices] + users_emb_final1
    api_prec_emb = api_pooled_output + items_emb_final

    loss = bpr_loss(users_emb_final1, users_emb_0, pos_items_emb_final, pos_items_emb_0, neg_items_emb_final,
                    neg_items_emb_0, lambda_val)
    cl_loss_m = CL_loss(mashup_cl_emb, users_emb_final1, temperature=temp)
    cl_loss_a = CL_loss(api_cl_emb, pos_items_emb_final, temperature=temp)

    precision_average, recall_average, f1_score_average, MAP_average, ndcg_average = my_get_metrics_3(sanfm_model, k,
                                                                                                      batch_size,
                                                                                                      mashup_perc_emb,
                                                                                                      api_prec_emb,
                                                                                                      test_label)

    return precision_average, recall_average, f1_score_average, MAP_average, ndcg_average, loss.item(), cl_loss_m.item(), cl_loss_a.item()



def my_evaluation_3_remove_BERT(model, edge_index, test_edge_index, test_sparse_edge_index, k, lambda_val, mashup_emb, api_emb,
                    temp, sanfm_model, batch_size, label):
    """Evaluates model loss and metrics including recall, precision, ndcg @ k

    Args:
        model (LighGCN): lightgcn model
        edge_index (torch.Tensor): 2 by N list of edges for split to evaluate
        sparse_edge_index (sparseTensor): sparse adjacency matrix for split to evaluate
        exclude_edge_indices ([type]): 2 by N list of edges for split to discount from evaluationolll
        k (int): determines the top k items to compute metrics on
        lambda_val (float): determines lambda for bpr loss

    Returns:
        tuple: bpr loss, recall @ k, precision @ k, ndcg @ k
    """

    # get embeddings

    user_indices, pos_item_indices, neg_item_indices = structured_negative_sampling(test_edge_index,
                                                                                    contains_neg_self_loops=False,
                                                                                    num_nodes=test_edge_index[
                                                                                                  1].max().item() + 1)

    users_emb_final, users_emb_0, items_emb_final, items_emb_0, api_pooled_output, mashup_pooled_output = model(
        test_sparse_edge_index, mashup_emb, api_emb)

    # print(f"{max(neg_item_indices)=}")

    users_emb_final1, users_emb_0 = users_emb_final[user_indices], users_emb_0[user_indices]
    pos_items_emb_final, pos_items_emb_0 = items_emb_final[pos_item_indices], items_emb_0[pos_item_indices]
    neg_items_emb_final, neg_items_emb_0 = items_emb_final[neg_item_indices], items_emb_0[neg_item_indices]
    mashup_cl_emb = mashup_pooled_output[user_indices]
    api_cl_emb = api_pooled_output[pos_item_indices]

    my_test_set = edge_index[0, user_indices].tolist()

    test_label = [label[j] for j in my_test_set]

    mashup_perc_emb =  users_emb_final1
    api_prec_emb =  items_emb_final

    loss = bpr_loss(users_emb_final1, users_emb_0, pos_items_emb_final, pos_items_emb_0, neg_items_emb_final,
                    neg_items_emb_0, lambda_val)
    cl_loss_m = CL_loss(mashup_cl_emb, users_emb_final1, temperature=temp)
    cl_loss_a = CL_loss(api_cl_emb, pos_items_emb_final, temperature=temp)

    precision_average, recall_average, f1_score_average, MAP_average, ndcg_average = my_get_metrics_3(sanfm_model, k,
                                                                                                      batch_size,
                                                                                                      mashup_perc_emb,
                                                                                                      api_prec_emb,
                                                                                                      test_label)

    return precision_average, recall_average, f1_score_average, MAP_average, ndcg_average, loss.item(), cl_loss_m.item(), cl_loss_a.item()


def my_evaluation_3_remove_LGC(model, edge_index, test_edge_index, test_sparse_edge_index, k, lambda_val, mashup_emb, api_emb,
                    temp, sanfm_model, batch_size, label):
    """Evaluates model loss and metrics including recall, precision, ndcg @ k

    Args:
        model (LighGCN): lightgcn model
        edge_index (torch.Tensor): 2 by N list of edges for split to evaluate
        sparse_edge_index (sparseTensor): sparse adjacency matrix for split to evaluate
        exclude_edge_indices ([type]): 2 by N list of edges for split to discount from evaluationolll
        k (int): determines the top k items to compute metrics on
        lambda_val (float): determines lambda for bpr loss

    Returns:
        tuple: bpr loss, recall @ k, precision @ k, ndcg @ k
    """

    # get embeddings

    user_indices, pos_item_indices, neg_item_indices = structured_negative_sampling(test_edge_index,
                                                                                    contains_neg_self_loops=False,
                                                                                    num_nodes=test_edge_index[
                                                                                                  1].max().item() + 1)

    users_emb_final, users_emb_0, items_emb_final, items_emb_0, api_pooled_output, mashup_pooled_output = model(
        test_sparse_edge_index, mashup_emb, api_emb)

    # print(f"{max(neg_item_indices)=}")

    users_emb_final1, users_emb_0 = users_emb_final[user_indices], users_emb_0[user_indices]
    pos_items_emb_final, pos_items_emb_0 = items_emb_final[pos_item_indices], items_emb_0[pos_item_indices]
    neg_items_emb_final, neg_items_emb_0 = items_emb_final[neg_item_indices], items_emb_0[neg_item_indices]
    mashup_cl_emb = mashup_pooled_output[user_indices]
    api_cl_emb = api_pooled_output[pos_item_indices]

    my_test_set = edge_index[0, user_indices].tolist()

    test_label = [label[j] for j in my_test_set]

    mashup_perc_emb =  mashup_cl_emb
    api_prec_emb =  api_pooled_output

    loss = bpr_loss(users_emb_final1, users_emb_0, pos_items_emb_final, pos_items_emb_0, neg_items_emb_final,
                    neg_items_emb_0, lambda_val)
    cl_loss_m = CL_loss(mashup_cl_emb, users_emb_final1, temperature=temp)
    cl_loss_a = CL_loss(api_cl_emb, pos_items_emb_final, temperature=temp)

    precision_average, recall_average, f1_score_average, MAP_average, ndcg_average = my_get_metrics_3(sanfm_model, k,
                                                                                                      batch_size,
                                                                                                      mashup_perc_emb,
                                                                                                      api_prec_emb,
                                                                                                      test_label)

    return precision_average, recall_average, f1_score_average, MAP_average, ndcg_average, loss.item(), cl_loss_m.item(), cl_loss_a.item()

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


def my_get_metrics_2(model, exclude_edge_indices, k, batch_size, mashup_pooled_output, api_pooled_output, label):
    """Computes the evaluation metrics: recall, precision, and ndcg @ k

    Args:
        model (LighGCN): lightgcn model
        edge_index (torch.Tensor): 2 by N list of edges for split to evaluate
        exclude_edge_indices ([type]): 2 by N list of edges for split to discount from evaluation
        k (int): determines the top k items to compute metrics on

    Returns:
        tuple: recall @ k, precision @ k, ndcg @ k
    """

    best_precision_all = []
    best_recall_all = []
    best_f1_score_all = []
    best_MAP_all = []
    best_ndcg_all = []
    for a in range(10):
        indices = random.choices(exclude_edge_indices, k=batch_size)

        test_mashup_emb = mashup_pooled_output[indices]
        test_api_emp = api_pooled_output

        test_label = [label[j] for j in indices]
        mashup_expanded = test_mashup_emb[:, None, :].expand(-1, test_api_emp.shape[0], -1)  # 在第二个维度上进行广播
        api_expanded = test_api_emp[None, :, :].expand(test_mashup_emb.shape[0], -1, -1)  # 在第一个维度上进行广播

        # 拼接 mashup 和 api
        concatenated_emb = torch.cat((mashup_expanded, api_expanded), dim=2)

        # 将拼接后的向量 reshape 成你想要的形状 (num_specified_mashup, num_api, 2 * emb_dim)
        model_input = concatenated_emb.view(len(indices), api_pooled_output.shape[0], -1)
        # model_input 就是你的模型的输入
        # 将 model_input 分割成 n 个子数组
        sub_inputs = torch.chunk(model_input, batch_size, dim=0)
        precision_all = []
        recall_all = []
        f1_score_all = []
        MAP_all = []
        ndcg_all = []
        for epoch_index in range(len(sub_inputs)):
            b = sub_inputs[epoch_index].squeeze(0)
            out = model(b)

            _, recommended_items = torch.topk(out, k=k)
            # recommended_items, score = find_k_largest(N, out)
            actual_interests = [index for index, value in enumerate(test_label[epoch_index]) if value == 1]

            precision = My_metric.calculate_precision(actual_interests, recommended_items)
            recall = My_metric.calculate_recall(actual_interests, recommended_items)
            f1_score = My_metric.calculate_f1_score(precision, recall)
            MAP = My_metric.calculate_MAP(actual_interests, recommended_items, k)
            ndcg = My_metric.NDCG(actual_interests, recommended_items, k)
            precision_all.append(precision)
            recall_all.append(recall)
            f1_score_all.append(f1_score)
            MAP_all.append(MAP)
            ndcg_all.append(ndcg)
        precision_average = sum(precision_all) / len(precision_all)
        recall_average = sum(recall_all) / len(recall_all)
        f1_score_average = sum(f1_score_all) / len(f1_score_all)
        MAP_average = sum(MAP_all) / len(MAP_all)
        ndcg_average = sum(ndcg_all) / len(ndcg_all)

        best_precision_all.append(precision_average)
        best_recall_all.append(recall_average)
        best_f1_score_all.append(f1_score_average)
        best_MAP_all.append(MAP_average)
        best_ndcg_all.append(ndcg_average)
        # print(
        #     "Average Precision: {:.5f}, Average Recall: {:.5f}, Average F1 Score: {:.5f}, Average MAP: {:.5f}, Average NDCG: {:.5f}".format(
        #         precision_average, recall_average, f1_score_average, MAP_average, ndcg_average))
    precision_average = sum(best_precision_all) / len(best_precision_all)
    recall_average = sum(best_recall_all) / len(best_recall_all)
    f1_score_average = sum(best_f1_score_all) / len(best_f1_score_all)
    MAP_average = sum(best_MAP_all) / len(best_MAP_all)
    ndcg_average = sum(best_ndcg_all) / len(best_ndcg_all)
    # print(
    #     "kkk Average Precision: {:.5f}, Average Recall: {:.5f}, Average F1 Score: {:.5f}, Average MAP: {:.5f}, Average NDCG: {:.5f}".format(
    #         precision_average, recall_average, f1_score_average, MAP_average, ndcg_average))
    return precision_average, recall_average, f1_score_average, MAP_average, ndcg_average


def my_get_metrics_3(model, k, batch_size, mashup, api, label):
    """Computes the evaluation metrics: recall, precision, and ndcg @ k

    Args:
        model (LighGCN): lightgcn model
        edge_index (torch.Tensor): 2 by N list of edges for split to evaluate
        exclude_edge_indices ([type]): 2 by N list of edges for split to discount from evaluation
        k (int): determines the top k items to compute metrics on

    Returns:
        tuple: recall @ k, precision @ k, ndcg @ k
    """

    indices = random.choices(range(mashup.shape[0]), k=batch_size)
    a=1

    test_mashup_emb = mashup[indices].to(device)
    test_api_emp = api.to(device)
    test_label = [label[i] for i in indices]

    mashup_expanded = test_mashup_emb[:, None, :].expand(-1, test_api_emp.shape[0], -1)  # 在第二个维度上进行广播
    api_expanded = test_api_emp[None, :, :].expand(test_mashup_emb.shape[0], -1, -1)  # 在第一个维度上进行广播

    # 拼接 mashup 和 api
    concatenated_emb = torch.cat((mashup_expanded, api_expanded), dim=2)

    # 将拼接后的向量 reshape 成你想要的形状 (num_specified_mashup, num_api, 2 * emb_dim)
    model_input = concatenated_emb.view(batch_size, test_api_emp.shape[0], -1)
    # model_input 就是你的模型的输入
    # 将 model_input 分割成 n 个子数组
    sub_inputs = torch.chunk(model_input, batch_size, dim=0)
    # 使用test_mashup_emb中的数据
    precision_all = []
    recall_all = []
    f1_score_all = []
    MAP_all = []
    ndcg_all = []
    for i in range(len(sub_inputs)):
        b = sub_inputs[i].squeeze(0)
        out = model(b)
        out.to(device)
        _, recommended_items = torch.topk(out, k=k)
        actual_interests = [index for index, value in enumerate(test_label[i]) if value == 1]
        precision = My_metric.calculate_precision(actual_interests, recommended_items)
        recall = My_metric.calculate_recall(actual_interests, recommended_items)
        f1_score = My_metric.calculate_f1_score(precision, recall)
        MAP = My_metric.calculate_MAP(actual_interests, recommended_items, k)
        ndcg = My_metric.NDCG(actual_interests, recommended_items, k)
        precision_all.append(precision)
        recall_all.append(recall)
        f1_score_all.append(f1_score)
        MAP_all.append(MAP)
        ndcg_all.append(ndcg)
    precision_average = sum(precision_all) / len(precision_all)
    recall_average = sum(recall_all) / len(recall_all)
    f1_score_average = sum(f1_score_all) / len(f1_score_all)
    MAP_average = sum(MAP_all) / len(MAP_all)
    ndcg_average = sum(ndcg_all) / len(ndcg_all)

    # print(
    #     "kkk Average Precision: {:.5f}, Average Recall: {:.5f}, Average F1 Score: {:.5f}, Average MAP: {:.5f}, Average NDCG: {:.5f}".format(
    #         precision_average, recall_average, f1_score_average, MAP_average, ndcg_average))
    return precision_average, recall_average, f1_score_average, MAP_average, ndcg_average


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