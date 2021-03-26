# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 16:54:06 2021

@author: Anna
"""
import hnswlib
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.semi_supervised import LabelPropagation

def cal_ber (data, label, s=10):
    ###build index for overall dataset
    p = hnswlib.Index(space = 'l2', dim = data.shape[1])
    p.init_index(max_elements = len(data), ef_construction = 200, M = 16)
    p.add_items(data)
    p.set_ef(50)
    ###
    cluster_list = []     #homogeneous cluster list
    cluster_centers = []  #homogeneous cluster centers list
    cluster_temp = [np.arange(len(label))]
    while cluster_temp:
        ind_cur = cluster_temp[0]
        if len(ind_cur)>s:
            kmeans = MiniBatchKMeans(n_clusters=2, random_state=0, batch_size=1000).fit(data[ind_cur,:])
            cluster1_ind = ind_cur[np.where(kmeans.labels_==0)[0]]
            cluster2_ind = ind_cur[np.where(kmeans.labels_==1)[0]]
            cluster1, cluster2 = kmeans.cluster_centers_
        
            if len(set(label[cluster1_ind].ravel()))==1:
                cluster_list.append(cluster1_ind)
                cluster_centers.append(cluster1)
            else:
                if len(cluster1_ind)>s:
                    cluster_temp.append(cluster1_ind)
            if len(set(label[cluster2_ind].ravel()))==1:
                cluster_list.append(cluster2_ind)
                cluster_centers.append(cluster2)
            else:
                if len(cluster2_ind)>s:
                    cluster_temp.append(cluster2_ind)
            del cluster_temp[0]
        else:
            del cluster_temp[0]
    # s可以减小，但是得判别得到类簇得置信程度，拟采用得方式是聚类
    confi_index = []
    cencter_label = []
    cencter_matrix = np.zeros((len(cluster_centers),data.shape[1]))
    for i in range(len(cluster_list)):
        confi_index.extend(cluster_list[i])
        cencter_matrix[i,:] = cluster_centers[i]
        cencter_label.append(label[cluster_list[i][0]][0])
    nonconfi_index = list(set(np.arange(len(label)))-set(confi_index))
    ### employ clusting method to recgonize cluster
    D = []
    L = []
    clustering = AgglomerativeClustering().fit(cencter_matrix)
    pre_label = clustering.labels_
    for i in range(len(set(pre_label))):
        i_index = np.where(pre_label==i)[0]
        temp = []
        for j in i_index:
            temp.extend(cluster_list[j])
        D.append(temp) 
        L.append(label[temp[0]][0])
    ### first round distillation--Knn query
    new_confi = []
    for i in range(len(L)):
        knn_ind = p.knn_query(data[D[i],:], k = int(len(label)/1000))[0]
        new_recog = list(set(nonconfi_index) & set(knn_ind.ravel()))
        new_confi.extend(np.where(label[new_recog]==L[i])[0])
        nonconfi_index = list(set(nonconfi_index)-set(new_confi))
    ### second round distillation--semi-superived Label Propagation
    model = LabelPropagation()
    data_con = np.concatenate((cencter_matrix,data[nonconfi_index,:]))
    label_con = np.concatenate((cencter_label,label[nonconfi_index].ravel()))
    model.fit(data_con, label_con)
    yhat = model.predict(data[nonconfi_index,:])
    BER = len(np.where(label[nonconfi_index].ravel()-yhat !=0)[0])/(len(label))
    return BER
