from google.cloud import bigquery
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import GraphSAGE
from torch_geometric.data import Data
import time
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.cluster import KMeans
from torcheval.metrics.functional import *
import torcheval.metrics as metrics
from matplotlib import cbook


# Get nodes tables from Google Cloud BigQuery
def nodes_tables_from_bq(bq):
    query_U = """  
        SELECT * 
        FROM `gannett-datascience.Ying.p4_node_user` 
        """
    query_Ut = """  
        SELECT * 
        FROM `gannett-datascience.Ying.p4_node_user_day` 
        """
    query_C = """  
        SELECT * 
        FROM `gannett-datascience.Ying.p4_node_content` 
        """
    query_E = """  
        SELECT * 
        FROM `gannett-datascience.Ying.p4_node_entities` 
        """
    query_job = bq.query(query_U, location="US")
    node_user = query_job.to_dataframe()
    query_job = bq.query(query_Ut, location="US")
    node_user_day = query_job.to_dataframe()
    query_job = bq.query(query_C, location="US")
    node_content = query_job.to_dataframe()
    query_job = bq.query(query_E, location="US")
    node_entities = query_job.to_dataframe()
    return node_user, node_user_day, node_content, node_entities


# use local csv files when possible
def load_nodes_tables(bq, local=True):
    # exists_user = os.path.isfile('data/node_user.csv')
    # exists_content = os.path.isfile('data/node_content.csv')
    # exists_entities = os.path.isfile('data/node_entities.csv')
    if local:
        print("Read from local csv files.")
        node_user = pd.read_csv('data/node_user.csv')
        node_user.drop('Unnamed: 0', axis=1, inplace=True)
        node_user_day = pd.read_csv('data/node_user_day.csv')
        node_user_day.drop('Unnamed: 0', axis=1, inplace=True)
        node_content = pd.read_csv('data/node_content.csv')
        node_content.drop('Unnamed: 0', axis=1, inplace=True)
        node_entities = pd.read_csv('data/node_entities.csv')
        node_entities.drop('Unnamed: 0', axis=1, inplace=True)
    else:
        print("Read from BigQuery.")
        node_user, node_user_day, node_content, node_entities = nodes_tables_from_bq(bq)
        node_user.to_csv('data/node_user.csv')
        node_user_day.to_csv('data/node_user_day.csv')
        node_content.to_csv('data/node_content.csv')
        node_entities.to_csv('data/node_entities.csv')

    return node_user, node_user_day, node_content, node_entities


def edge_indices_from_bq(bq):
    user_ut = """  
        SELECT * 
        FROM `gannett-datascience.Ying.p5_edge_user_ut` 
        """
    ut_content = """  
        SELECT * 
        FROM `gannett-datascience.Ying.p5_edge_ut_content` 
        """
    content_entity = """  
        SELECT * 
        FROM `gannett-datascience.Ying.p5_edge_content_entities` 
        """
    entities = """  
        SELECT * 
        FROM `gannett-datascience.Ying.p5_edge_entities` 
        """

    query_job = bq.query(user_ut, location="US")
    u_to_ut = query_job.to_dataframe()
    query_job = bq.query(ut_content, location="US")
    ut_to_c = query_job.to_dataframe()
    query_job = bq.query(content_entity, location="US")
    c_to_e = query_job.to_dataframe()
    query_job = bq.query(entities, location="US")
    e_to_e = query_job.to_dataframe()

    return u_to_ut, ut_to_c, c_to_e, e_to_e


# use local csv files when possible
def load_edge_indices(bq, local=True):
    # exists_u_c = os.path.isfile('data/edge_index_user_to_content.csv')
    # exists_c_e = os.path.isfile('data/edge_index_content_to_entity.csv')
    # exists_e_e = os.path.isfile('data/edge_index_entity_to_entity.csv')
    if local:
        print("Read from local csv files.")
        u_to_ut = pd.read_csv('data/edge_index_user_to_ut.csv')
        u_to_ut.drop('Unnamed: 0', axis=1, inplace=True)
        ut_to_c = pd.read_csv('data/edge_index_ut_to_content.csv')
        ut_to_c.drop('Unnamed: 0', axis=1, inplace=True)
        c_to_e = pd.read_csv('data/edge_index_content_to_entity.csv')
        c_to_e.drop('Unnamed: 0', axis=1, inplace=True)
        e_to_e = pd.read_csv('data/edge_index_entity_to_entity.csv')
        e_to_e.drop('Unnamed: 0', axis=1, inplace=True)
    else:
        print("Read from BigQuery.")
        u_to_ut, ut_to_c, c_to_e, e_to_e = edge_indices_from_bq(bq)
        u_to_ut.to_csv('data/edge_index_user_to_ut.csv')
        ut_to_c.to_csv('data/edge_index_ut_to_content.csv')
        c_to_e.to_csv('data/edge_index_content_to_entity.csv')
        e_to_e.to_csv('data/edge_index_entity_to_entity.csv')

    return u_to_ut, ut_to_c, c_to_e, e_to_e


def cal_accuracy(preds, labels, correct_count, all_count):
    for i in range(len(labels)):
        logps = preds[i]
        # Output of the network are log-probabilities, need to take exponential for probabilities
        probab = list(torch.exp(logps))
        pred_label = probab.index(max(probab))
        true_label = labels.numpy()[i]
        if (true_label == pred_label):
            correct_count += 1
        all_count += 1
    return correct_count, all_count


def train(model1, model2, optimizer, train_loader, device, loss_weight=(0.5, 0.5)):  # user_node_id_offset, data,
    model1.train()
    model2.train()
    total_loss = 0
    correct_count, all_count = 0, 0
    ep_loss1, ep_loss2 = 0.0, 0.0

    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        # SAGE
        h = model1(batch.x, batch.edge_index, batch.edge_weight)
        h_src = h[batch.edge_label_index[0]]
        h_dst = h[batch.edge_label_index[1]]
        pred = (h_src * h_dst).sum(dim=-1)
        loss1 = F.binary_cross_entropy_with_logits(pred, batch.edge_label)/1000000.0

        # user nodes embeddings and y in the batch
        mask = batch.x[:, 7] == 1
        mask = torch.reshape(mask.nonzero(), (-1,))
        fea = h[mask, :]
        label = batch.y[mask]
        assert fea.size()[0] == label.size()[0]
        assert fea.size()[0] == batch.x[:, 7].sum()
        assert label.max() == 1

        if mask.shape[0] > 0:
            # Fully Connect NN
            train_pred = model2(fea)
            train_pred = torch.reshape(train_pred, (-1,))
            bcel = nn.BCEWithLogitsLoss()
            loss2 = bcel(input=train_pred, target=label)
        else:
            loss2 = 0.0

        loss = loss_weight[0] * loss1 + loss_weight[1] * loss2
        # Calculate train accuracy
        # correct_count, all_count = cal_accuracy(train_pred, label, correct_count, all_count)
        loss.backward()
        optimizer.step()

        ep_loss1 += loss1
        ep_loss2 += loss2
        # total_loss += float(loss) * pred.size(0)  # ??

    # print(binary_confusion_matrix(train_pred[:,1], label, threshold=0.5))
    # print(metrics.BinaryF1Score(threshold=0.5).update())
    # print()
    return ep_loss1, ep_loss2  # loss  # total_loss / data.num_nodes


# Save a trained model
def save_model(model, path):
    if os.path.isfile(path):
        os.remove(path)
    torch.save(model.state_dict(), path)


# Load a model
def load_model(model, path):
    model.load_state_dict(torch.load(path))


# Hosts from the top 50 news sites
def top_hosts_from_bq(bq):
    top_host_query = """  
        SELECT * 
        FROM `gannett-datarevenue.zz_test_pang.top_hosts` 
        """
    query_job = bq.query(top_host_query, location="US")
    top_hosts = query_job.to_dataframe()

    return top_hosts


# use local csv files when possible
def load_top_hosts(bq):
    exists = os.path.isfile('data/top_hosts.csv')
    if exists:
        print("Read from local csv files.")
        top_hosts = pd.read_csv('data/top_hosts.csv')
    else:
        print("Read from BigQuery.")
        top_hosts = top_hosts_from_bq(bq)
        top_hosts.to_csv('data/top_hosts.csv')

    return top_hosts


def num_boxPlot_desc(data, num_column, figsize=(8, 5), title=None):
    fig = plt.figure(figsize=figsize)

    ax1 = fig.add_subplot(121)
    data.boxplot(column=[num_column])
    Qs = cbook.boxplot_stats(data[num_column])
    iqr = Qs[0]['q3'] - Qs[0]['q1']
    for v in [Qs[0][i] for i in ['q1', 'med', 'q3']] + [Qs[0]['q1'] - 1.5 * iqr, Qs[0]['q3'] + 1.5 * iqr]:
        ax1.text(1.1, v, str(round(v, 1)), verticalalignment='center')

    ax2 = fig.add_subplot(122)
    df = data[[num_column]].describe().apply(lambda s: s.apply('{0:.5f}'.format))
    font_size = 10
    bbox = [0.3, 0, 0.6, 0.8]
    ax2.axis('off')
    mpl_table = ax2.table(cellText=df.values, rowLabels=df.index, bbox=bbox, colLabels=df.columns)
    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)
    plt.title(title)
