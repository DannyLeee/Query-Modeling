#%%
import numpy as np
from collections import Counter
from tqdm import tqdm
from datetime import datetime,timezone,timedelta
from functools import reduce
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import pickle
from scipy import sparse
from scipy.sparse import coo_matrix
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
args = dotdict(A=5, B=1, scratch=0)

# bm_B = args.bm_B
# K1 = args.K1
# K3 = args.K3
# KL_A = args.KL_A
# KL_B = args.KL_B
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
doc_len = []
for txt in tqdm(file_iter("q")):
    list_q_tf += [Counter(txt.split())]
    query_list += [txt]

for txt in tqdm(file_iter("d")):
    list_d_tf += [Counter(txt.split())]
    doc_list += [txt]
    doc_len += [len(txt.split())]
doc_avg_len = np.array(doc_len).mean()
total_len = sum(doc_len)

#%%
# voc
if args.scratch:
    q_voc = reduce(set.union, map(set, map(dict.keys, list_q_tf)))
    q_voc = list(q_voc)
    voc = reduce(set.union, map(set, map(dict.keys, list_d_tf)))
    voc = list(voc)
else:
    with open("model/q_voc.pkl", "rb") as fp:
        q_voc = pickle.load(fp)
    with open("model/voc.pkl", "rb") as fp:
        voc = pickle.load(fp)

#%%
def counter2array(voc, li, path, list_of_counter):
    row = []
    col = []
    data = []
    for i, w_i in tqdm(enumerate(voc)):
        row += [i]
        col += [0]
        data += [0]
        for j in range(len(li)):
            if list_of_counter[j][w_i] != 0:
                row += [i]
                col += [j]
                data += [list_of_counter[j][w_i]]
    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    tf_array = coo_matrix((data, (row, col)))
    sparse.save_npz(path, tf_array)
    del row,col,data
    return tf_array

#%%
if args.scratch:
    tf_array = counter2array(voc, d_list, 'model/tf_array.npz', list_d_tf)
else:
    tf_array = sparse.load_npz("model/tf_array.npz")

#%%
# df
if args.scratch:
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
    with open("model/bm_df.pkl", "wb") as fp:
        pickle.dump(bm_df, fp)
    np.save("model/BG.npy", BG)
else:
    with open("model/bm_df.pkl", "rb") as fp:
        bm_df = pickle.load(fp)
    BG = np.load("model/BG.npy")

#%%
# small voc
if args.scratch:
    tf_array = tf_array.tocsr()
    small_tf_array = []
    small_voc = []
    small_BG = []
    for arg in BG.argsort()[-10000: ]:
        small_tf_array += [tf_array[arg]]
        small_voc += [voc[arg]]
        small_BG += [BG[arg]]
    small_tf_array = sparse.vstack([row for row in small_tf_array]).toarray()
    small_BG = np.array(small_BG)

    with open("model/small_voc.pkl", "wb") as fp:
        pickle.dump(small_voc, fp)
    np.save("model/small_tf_array.npy", small_tf_array)
    q_tf_array = counter2array(small_voc, q_list, 'model/q_tf_array.npz', list_q_tf)
    np.save("model/small_BG.npy", small_BG)
else:
    with open("model/small_voc.pkl", "rb") as fp:
        small_voc = pickle.load(fp)
    small_tf_array = np.load("model/small_tf_array.npy")
    q_tf_array = sparse.load_npz('model/q_tf_array.npz')
    small_BG = np.load("model/small_BG.npy")

#%%
#idf
idf = np.log(1+(len(d_list)+1) / (np.count_nonzero(small_tf_array, axis=1)+1))

#%%
# tfidf
np_q_tfidf = (1 + np.ma.log2(q_tf_array.toarray())).transpose() * idf
np_d_tfidf = (1 + np.ma.log2(small_tf_array)).transpose() * idf
# del np_q_tf, np_d_tf

np_q_tfidf = np_q_tfidf.filled(0)
np_d_tfidf = np_d_tfidf.filled(0)

#%%
from sklearn.metrics.pairwise import cosine_similarity
sim_array = cosine_similarity(np_q_tfidf, np_d_tfidf)
sim_array = np.array(sim_array)
#========== VSM ==========#

#%%
new_q_tfidf = np.zeros(len(small_voc))
for q in tqdm(range(len(q_list))):
    relate_doc_vac = []
    relate_arg = np.argsort(sim_array[q])[-5:]
    relate_doc_vac = np.vstack([np_d_tfidf[_, :] for _ in relate_arg])
    relate_doc_vac = relate_doc_vac.sum(0)
    new_q_vac = A*np_q_tfidf[q] + B*relate_doc_vac/5
    new_q_tfidf = np.vstack((new_q_tfidf, new_q_vac))
new_q_tfidf = new_q_tfidf[1:]
sim_array = cosine_similarity(new_q_tfidf, np_d_tfidf)
sim_array = np.array(sim_array)

#%%
# output
with open('result.csv', 'w') as output_file:
    output_file.write("Query,RetrievedDocuments\n")
    for i, q_id in tqdm(enumerate(q_list)):
        output_file.write(q_id+',')
        sorted = np.argsort(sim_array[i])[-5000:]
        sorted = np.flip(sorted)
        for _, j in enumerate(sorted):
            output_file.write(d_list[j]+' ')
        output_file.write('\n')
timestamp()
# %%
