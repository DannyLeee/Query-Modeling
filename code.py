#%%
import numpy as np
from collections import Counter
from tqdm import tqdm
from datetime import datetime,timezone,timedelta
from functools import reduce
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from scipy import sparse
from scipy.sparse import csr_matrix
from numba import jit

def timestamp():
    dt1 = datetime.utcnow().replace(tzinfo=timezone.utc)
    dt2 = dt1.astimezone(timezone(timedelta(hours=8))) # 轉換時區 -> 東八區
    print(dt2)

def file_iter(_type):
    if _type == "q":
        for name in q_list:
            with open(query_path+name+'.txt') as f:
                yield f.readline()
    elif _type == "d":
        for name in d_list:
            with open(doc_path+name+'.txt') as f:
                yield f.readline()

# parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
# parser.add_argument("-B", default=0.7, type=float ,help=" ")
# parser.add_argument("-K1", default=0.8, type=float ,help=" ")
# parser.add_argument("-K3", default=-1, type=float ,help="if K3 =-1 not use query TF")
# args = parser.parse_args()

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
args = dotdict(B=0.7, K1=0.8, K3=-1)

B = args.B
K1 = args.K1
K3 = args.K3

#%%
timestamp()
doc_path = "data/docs/"
query_path = "data/queries/"

d_list = []
with open('data/doc_list.txt', 'r') as d_list_file:
    for line in d_list_file:
        line = line.replace("\n", "")
        d_list += [line]

q_list = []
with open('data/query_list.txt', 'r') as q_list_file:
    for line in q_list_file:
        line = line.replace("\n", "")
        q_list += [line]
# tf
list_q_tf = []
list_d_tf = []
doc_list = []
query_list = []
for txt in tqdm(file_iter("q")):
    list_q_tf += [Counter(txt.split())]
    query_list += [txt]

for txt in tqdm(file_iter("d")):
    list_d_tf += [Counter(txt.split())]
    doc_list += [txt]

#%%
# voc
voc = reduce(set.union, map(set, map(dict.keys, list_q_tf)))
voc = list(voc)

#%%
row = []
col = []
data = []
for i, w_i in tqdm(enumerate(voc)):
    for j in range(len(d_list)):
        if list_d_tf[j][w_i] != 0:
            row += [i]
            col += [j]
            data += [list_d_tf[j][w_i]]
row = np.array(row)
col = np.array(col)
data = np.array(data)
#%%
from scipy import sparse
from scipy.sparse import csr_matrix
tf_array = csr_matrix((data, (row, col)))
sparse.save_npz('model/tf_array.npz', tf_array)
del row,col,data

#%%
import pickle
with open("model/q_voc.pkl", "wb") as fp:
    pickle.dump(voc, fp)
#%%
with open("model/query_counter.pkl", "wb") as fp:
    pickle.dump(list_q_tf, fp)
with open("model/doc_counter.pkl", "wb") as fp:
    pickle.dump(list_d_tf, fp)

#%%
# df
with open("model/voc.pkl", "rb") as fp:
    voc = pickle.load(fp)
doc_len = []
np_df = dict.fromkeys(voc, 0)
for i, doc in tqdm(enumerate(doc_list)):
    doc_len += [len(doc)]
    for w in voc:   
        np_df[w] += 1 if list_d_tf[i][w] > 0 else 0
doc_avg_len = np.array(doc_len).mean()

#%%
np.save("model/query/np_df.npy", np_df)

#%%
# sim_array
sim_array = np.zeros([len(q_list), len(d_list)])
for q in tqdm(range(len(q_list))):
    for d in range(len(d_list)):
        for w in query_list[q].split():
            d_tf = (K1+1)*list_d_tf[d][w] / (K1*((1-B) + B*doc_len[d]/doc_avg_len) + list_d_tf[d][w])
            q_tf = (K3+1)*list_q_tf[q][w] / (K3+list_q_tf[q][w]) if K3 != -1 else 1
            idf = np.log(1+(len(d_list)-np_df[w]+0.5) / (np_df[w]+0.5))
            idf = np.power(idf, 2)
            sim_array[q][d] += d_tf * q_tf * idf

#=================== BM25 ===================#



#%%
# output
with open('result.csv', 'w') as output_file:
    output_file.write("Query,RetrievedDocuments\n")
    for i, q_id in tqdm(enumerate(q_list)):
        output_file.write(q_id+',')
        sorted = np.argsort(sim_array[i])
        sorted = np.flip(sorted)[:5000]
        for _, j in enumerate(sorted):
            output_file.write(d_list[j]+' ')
        output_file.write('\n')
timestamp()