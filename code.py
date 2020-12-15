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

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("-bm_B", metavar="bm_beta", default=0.75, type=float ,help="BM model beta")
parser.add_argument("-K1", default=3.5, type=float ,help="BM model K1")
parser.add_argument("-K3", default=-1, type=float ,help="BM model K3; if -1 not use query TF")
parser.add_argument("-A", metavar="alpha", default=1, type=float, help=" ")
parser.add_argument("-B", metavar="beta", default=0.5, type=float, help=" ")
parser.add_argument("-C", metavar="gamma", default=0.15, type=float,help=" ")
parser.add_argument("-r_doc", default=5, type=int, help="the number of related documents")
parser.add_argument("-nr_doc", default=1, type=int, help="the number of non-related documents")
parser.add_argument("-step", default=5, type=int, help="Rocchio iteration step")
parser.add_argument("-scratch", default=0, type=int, help=" ")
parser.add_argument("-save", default=0, type=int, help=" ")
args = parser.parse_args()

#%%
class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
# args = dotdict(bm_B=0.75, K1=3.5, K3=-1, A=1, B=0.5, C=0.15, step=5, r_doc=5, nr_doc=1, scratch=0, save=0)
bm_B = args.bm_B
K1 = args.K1
K3 = args.K3
A = args.A
B = args.B
C = args.C
r_doc = args.r_doc
nr_doc = args.nr_doc

#%%
timestamp()
# root_path = "../HW1 Vector Space Model/"
root_path = "./"
doc_path = root_path+"data/docs/"
query_path = root_path+"data/queries/"

d_list = []
with open(root_path+'data/doc_list.txt', 'r') as d_list_file:
    for line in d_list_file:
        line = line.replace("\n", "")
        d_list += [line]

q_list = []
with open(root_path+'data/query_list.txt', 'r') as q_list_file:
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

if args.scratch:
    for txt in tqdm(file_iter("d")):
        list_d_tf += [Counter(txt.split())]
        doc_list += [txt]
        doc_len += [len(txt.split())]
    if args.save:
        with open("model/list_d_tf.pkl", "wb") as fp:
            pickle.dump(list_d_tf, fp)
        with open("model/doc_list.pkl", "wb") as fp:
            pickle.dump(doc_list, fp)
        with open("model/doc_len.pkl", "wb") as fp:
            pickle.dump(doc_len, fp)
else:
    with open("model/list_d_tf.pkl", "rb") as fp:
        list_d_tf = pickle.load(fp)
    with open("model/doc_list.pkl", "rb") as fp:
        doc_list = pickle.load(fp)
    with open("model/doc_len.pkl", "rb") as fp:
        doc_len = pickle.load(fp)

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
        for j in range(len(li)):
            if list_of_counter[j][w_i] != 0:
                row += [i]
                col += [j]
                data += [list_of_counter[j][w_i]]
    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    tf_array = coo_matrix((data, (row, col)), shape=(len(voc), len(li)))
    if args.save:
        sparse.save_npz(path, tf_array)
    del row,col,data
    return tf_array

#%%
# tf array
if args.scratch:
    tf_array = counter2array(voc, d_list, 'model/tf_array.npz', list_d_tf)
else:
    tf_array = sparse.load_npz("model/tf_array.npz")

#%%
# df
if args.scratch:
    with open("model/voc.pkl", "rb") as fp:
        voc = pickle.load(fp)
    bm_df = dict.fromkeys(voc, 0)
    BG = [0] * len(voc)
    for j, doc in tqdm(enumerate(doc_list)):
        for i, w in enumerate(voc):
            bm_df[w] += 1 if list_d_tf[j][w] > 0 else 0
            BG[i] += list_d_tf[j][w]
    BG = np.array(BG)
    if args.save:
        with open("model/bm_df.pkl", "wb") as fp:
            pickle.dump(bm_df, fp)
        np.save("model/BG.npy", BG)
else:
    with open("model/bm_df.pkl", "rb") as fp:
        bm_df = pickle.load(fp)
    BG = np.load("model/BG.npy")

#%%
#idf
tf_array = tf_array.tocsr()
n_i = tf_array.indptr[1:] - tf_array.indptr[:-1]
idf = 1 + np.log((len(d_list)+1) / (n_i+1)) # ndarray

#%%
q_tf_array = counter2array(voc, q_list, 'model/q_tf_array.npz', list_q_tf)
q_tf_array = q_tf_array.transpose()
tf_array = tf_array.transpose()

#%%
# tf log norm
q_tf_array.data = 1 + np.log(q_tf_array.data)
tf_array.data = 1 + np.log(tf_array.data)
# tf*idf
np_q_tfidf = q_tf_array.multiply(idf).tocsr()
np_d_tfidf = tf_array.multiply(idf).tocsr()

#%%
# tfidf norm
if args.scratch:
    for i, row in tqdm(enumerate(np_d_tfidf)):
        np_d_tfidf[i] /= (row*row.T).power(0.5).data[0]
else:
    np_d_tfidf = sparse.load_npz("model/d_tfidf.npz").tocsr()

for i, row in tqdm(enumerate(np_q_tfidf)):
    np_q_tfidf[i] /= (row*row.T).power(0.5).data[0]

#%%
# bm_sim_array
if args.scratch:
    bm_sim_array = np.zeros([len(q_list), len(d_list)])
    for q in tqdm(range(len(q_list))):
        for d in range(len(d_list)):
            for w in query_list[q].split():
                d_tf = (K1+1)*list_d_tf[d][w] / (K1*((1-bm_B) + bm_B*doc_len[d]/doc_avg_len) + list_d_tf[d][w])
                q_tf = (K3+1)*list_q_tf[q][w] / (K3+list_q_tf[q][w]) if K3 != -1 else 1
                idf = np.log(1+(len(d_list)-bm_df[w]+0.5) / (bm_df[w]+0.5))
                idf = np.power(idf, 2)
                bm_sim_array[q][d] += d_tf * q_tf * idf
    np.save("model/bm_sim.npy", bm_sim_array)
else:
    bm_sim_array = np.load("model/bm_sim.npy")

#%%
sim_array = bm_sim_array

#%%
from sklearn.metrics.pairwise import cosine_similarity
for step in range(args.step):
    new_q_tfidf = sparse.csr_matrix((1,len(voc)))
    for q in tqdm(range(len(q_list))):
        relate_arg = np.argsort(sim_array[q])
        relate_doc_vec = sparse.vstack([np_d_tfidf[_, :] for _ in relate_arg[-r_doc:]])
        relate_doc_vec = relate_doc_vec.mean(0)
        non_relate_doc_vec = sparse.vstack([np_d_tfidf[_, :] for _ in relate_arg[:nr_doc]])
        non_relate_doc_vec = non_relate_doc_vec.mean(0)
        new_q_vec = A*np_q_tfidf[q] + B*relate_doc_vec - C*non_relate_doc_vec
        new_q_tfidf = sparse.vstack((new_q_tfidf, new_q_vec)).tocsr()
    new_q_tfidf = new_q_tfidf[1:]
    sim_array = new_q_tfidf*np_d_tfidf.T # row-wise dot product = matrix multiply
    sim_array = sim_array.toarray()
    np_q_tfidf = new_q_tfidf

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