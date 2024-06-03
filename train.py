from dataset import DataLoad
from torch_sparse import SparseTensor
from sklearn.model_selection import train_test_split
from model import Model_Multiple
from sanfm import SANFM, snafm_loss
from utils import bpr_loss, my_sample_mini_batch, CL_loss, ModelConfig, normalize_sample_label,my_evaluation_3
import torch
from tqdm import tqdm

from itertools import chain

def train(sanfm_emb,lgc_emb,cl_rate,temp,dropout):
    print("开始训练")
    config = ModelConfig()
    myseed = 3030
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mashup_mapping,api_mapping,edge_index,mashup_emb,api_emb=DataLoad("Train")
    mashup_mapping_test, api_mapping_test, edge_index_test, mashup_emb_test, api_emb_test, test_label_test = DataLoad("Test")

    num_mashup = len(mashup_mapping)
    num_api = len(api_mapping)
    num_edges = len(edge_index[1])
    all_index = [i for i in range(num_edges)]

    train_index, test_index=train_test_split(all_index ,test_size=0.2, random_state=myseed)
    train_edge_index = edge_index[:, train_index]
    test_edge_index = edge_index[:, test_index]

    train_sparse_edge_index = SparseTensor(row=train_edge_index[0], col=train_edge_index[1],
                                           sparse_sizes=(num_api + num_mashup, num_api + num_mashup)).to(device)
    test_sparse_edge_index = SparseTensor(row=test_edge_index[0], col=test_edge_index[1],
                                          sparse_sizes=(num_api + num_mashup, num_api + num_mashup)).to(device)

    model =Model_Multiple(num_mashup, num_api,embedding_dim=lgc_emb, num_layers=config.num_layers, add_self_loops=False,input_dim=config.input_dim,hidden_dim=lgc_emb)
    sanfm_obj = SANFM(embed_dim=sanfm_emb, droprate=dropout, i_num=lgc_emb*2)

    model = model.to(device)
    sanfm_obj=sanfm_obj.to(device)
    optimizer = torch.optim.Adam(params=chain(model.parameters(),sanfm_obj.parameters()), lr=config.lr, weight_decay=config.weight_decay)


    for epoch in tqdm(range(1, config.n_epoch + 1)):
        model.train()
        sanfm_obj.train()
        optimizer.zero_grad()
        user_indices, pos_item_indices, neg_item_indices, mashup_e, api_e, neg_api_e=my_sample_mini_batch(config.train_batch_size, train_edge_index,mashup_emb,api_emb )

        users_emb_final, users_emb_0, items_emb_final, items_emb_0,api_pooled_output, mashup_pooled_output, neg_api_pooled_output = model(train_sparse_edge_index,mashup_e, api_e, neg_api_e)
        #batch_size
        users_emb_final = users_emb_final[user_indices]
        users_emb_0 = users_emb_0[user_indices]
        pos_items_emb_final = items_emb_final[pos_item_indices]
        pos_items_emb_0 = items_emb_0[pos_item_indices]
        neg_items_emb_final = items_emb_final[neg_item_indices]
        neg_items_emb_0 = items_emb_0[neg_item_indices]
        train_loss = bpr_loss(users_emb_final, users_emb_0, pos_items_emb_final, pos_items_emb_0, neg_items_emb_final,
                              neg_items_emb_0, config.lamda)

        #batch_size_contrastive_learning
        cl_loss_m=CL_loss(mashup_pooled_output,users_emb_final,temperature=temp)
        cl_loss_a=CL_loss(api_pooled_output,pos_items_emb_final,temperature=temp)
        #BPR_loss

        #Batch Graph+Text
        mashup_emb_temp = mashup_pooled_output + users_emb_final
        api_emb_temp = api_pooled_output + pos_items_emb_final
        neg_api_emb_temp = neg_api_pooled_output + neg_items_emb_final

        sample,label=normalize_sample_label(mashup_emb_temp, api_emb_temp, neg_api_emb_temp)

        rec_loss = snafm_loss(sanfm_obj, sample, label)

        Totle_loss =rec_loss+train_loss + cl_rate*(cl_loss_a + cl_loss_m)
        # Totle_loss = rec_loss + train_loss
        Totle_loss.backward(retain_graph=True)
        optimizer.step()
        if (epoch % config.eval_steps == 0):
            model.eval()
            sanfm_obj.eval()
            precision_average, recall_average, f1_score_average, MAP_average, ndcg_average,loss,cl_loss_m,cl_loss_a = my_evaluation_3(model, edge_index,test_edge_index, test_sparse_edge_index,config.K,config.lamda,
                                                                     mashup_emb_test,api_emb_test,temp,sanfm_obj,config.test_batch_size,test_label_test)

            print(f"precision={precision_average:.5f}, recall={recall_average:.5f}, f1_score={f1_score_average:.5f}, MAP={MAP_average:.5f}, ndcg={ndcg_average:.5f}")


train(128,64,0.0005,0.001,0.5)
