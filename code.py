#%%
import numpy as np
from collections import Counter
from tqdm import tqdm
from datetime import datetime,timezone,timedelta
from functools import reduce
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import pickle
from scipy import sparse
from scipy.sparse import csc_matrix
from numba import jit

def timestamp(msg=""):
    dt1 = datetime.utcnow().replace(tzinfo=timezone.utc)
    dt2 = dt1.astimezone(timezone(timedelta(hours=8))) # 轉換時區 -> 東八區
    print(str(dt2)[:-13] + '\t' + msg)

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
args = dotdict(bm_B=0.7, K1=0.8, K3=-1, A=0.3, B=0.3, step=30, train_from=0)

bm_B = args.bm_B
K1 = args.K1
K3 = args.K3
A = args.A
B = args.B


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
with open("model/list_q_tf.pkl", "wb") as fp:
    pickle.dump(list_q_tf, fp)
with open("model/list_d_tf.pkl", "wb") as fp:
    pickle.dump(list_d_tf, fp)

#%%
# voc
q_voc = reduce(set.union, map(set, map(dict.keys, list_q_tf)))
q_voc = list(q_voc)
voc = reduce(set.union, map(set, map(dict.keys, list_d_tf)))
voc = list(voc)

#%%
with open("model/q_voc.pkl", "rb") as fp:
    q_voc = pickle.load(fp)
with open("model/voc.pkl", "rb") as fp:
    voc = pickle.load(fp)

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
tf_array = csc_matrix((data, (row, col)))
sparse.save_npz('model/tf_array.npz', tf_array)
del row,col,data

#%%
tf_array = sparse.load_npz("model/tf_array.npz")

#%%
# df
with open("model/voc.pkl", "rb") as fp:
    voc = pickle.load(fp)
doc_len = []
bm_df = dict.fromkeys(voc, 0)
BG = [0] * len(voc)
for j, doc in tqdm(enumerate(doc_list)):
    doc_len += [len(doc)]
    for i, w in enumerate(voc):   
        bm_df[w] += 1 if list_d_tf[j][w] > 0 else 0
        BG[i] += list_d_tf[j][w]
BG = np.array(BG)

#%%
doc_len = []
for j, doc in tqdm(enumerate(doc_list)):
    doc_len += [len(doc)]
doc_avg_len = np.array(doc_len).mean()
total_len = sum(doc_len)

#%%
with open("model/bm_df.pkl", "wb") as fp:
    pickle.dump(bm_df, fp)
np.save("model/BG.npy", BG)

#%%
with open("model/bm_df.pkl", "rb") as fp:
    bm_df = pickle.load(fp)
BG = np.load("model/BG.npy")

#%%
# sim_array
bm_sim_array = np.zeros([len(q_list), len(d_list)])
for q in tqdm(range(len(q_list))):
    for d in range(len(d_list)):
        for w in query_list[q].split():
            d_tf = (K1+1)*list_d_tf[d][w] / (K1*((1-bm_B) + bm_B*doc_len[d]/doc_avg_len) + list_d_tf[d][w])
            q_tf = (K3+1)*list_q_tf[q][w] / (K3+list_q_tf[q][w]) if K3 != -1 else 1
            idf = np.log(1+(len(d_list)-bm_df[w]+0.5) / (bm_df[w]+0.5))
            idf = np.power(idf, 2)
            bm_sim_array[q][d] += d_tf * q_tf * idf

#%%
np.save("model/bm_sim.npy", bm_sim_array)

#%%
bm_sim_array = np.load("model/bm_sim.npy")

#=================== BM25 ===================#

#%%
@jit(nopython=True)
def E_step(T_given_w, P_smm, BG, V, A, total_len):
    for i in range(V):
        T_given_w[i] = (1-A)*P_smm[i] / ((1-A)*P_smm[i] + A*BG[i]/total_len)
    return T_given_w

# @jit(nopython=True)
def M_step(tf_array, P_smm, T_given_w, V):
    for i in range(V):
        P_smm[i] = (tf_array[i]*T_given_w[i]).sum()
    P_smm /= P_smm.sum()
    return P_smm

#%%
sim_array = np.zeros([len(q_list), len(d_list)])
for q in range(len(q_list)):
    # initial
    T_given_w = np.zeros(len(voc))
    P_smm = np.random.rand(len(voc))
    P_smm /= P_smm.sum()
    log_likelihood = 0.0

    timestamp("relate_tf")
    relate_arg = np.argsort(bm_sim_array[q])[:1000] ###
    relate_tf = []
    for arg in relate_arg:
        relate_tf += [tf_array[:, arg]]
    relate_tf = sparse.hstack([col for col in relate_tf]).toarray()
    # relate_tf = np.array(relate_tf).transpose()

    timestamp("EM start")
    for step in (range(args.train_from, args.step)):
        # E-step
        # timestamp(f"\n{step+1}\t--E-step--")
        # timestamp("P(T_smm|w)")
        T_given_w = E_step(T_given_w, P_smm, BG, len(voc), A, total_len)

        # M-step
        # timestamp("--M-step--")
        # timestamp("P_smm(w)")
        P_smm = M_step(relate_tf, P_smm, T_given_w, len(voc))
        # timestamp("---")

        # TODO: log_likelihood

    timestamp("sim_arry")
    for d in tqdm(range(len(d_list))):
        for i, w in enumerate(voc):
            sim_array[q][d] += (A/3 + B* + (1-A-B)*BG[i]/total_len) * \
                list_d_tf[d][w]/doc_len[d]
sim_array *= -1

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