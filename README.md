## Introduction
* This is the Information Retrieval HW5
* Using Rocchio algorithm to implement pseudo-relevance feedback model that compute the relation between given querys and documents
    * By using Best Match model doing first retrieval 

## Usage
```
code.py [-h] [-bm_B bm_beta] [-K1 K1] [-K3 K3] [-A alpha] [-B beta]
               [-C gamma] [-r_doc R_DOC] [-nr_doc NR_DOC] [-scratch SCRATCH]
               [-save SAVE]

optional arguments:
  -h, --help        show this help message and exit
  -bm_B bm_beta     BM model beta (default: 0.75)
  -K1 K1            BM model K1 (default: 3.5)
  -K3 K3            BM model K3; if -1 not use query TF (default: -1)
  -A alpha          (default: 1)
  -B beta           (default: 0.5)
  -C gamma          (default: 0.15)
  -r_doc R_DOC      the number of related documents (default: 5)
  -nr_doc NR_DOC    the number of non-related documents (default: 1)
  -step STEP        Rocchio iteration step (default: 5)
  -scratch SCRATCH  (default: 0)
  -save SAVE        (default: 0)
```

## Approach
* About Best Match model
    * IDF different from original formula
        * Square IDF make it more important
        * <img src="https://latex.codecogs.com/gif.latex?IDF&space;=&space;(ln(1&space;&plus;&space;\frac{N&plus;0.5}{n_i&plus;0.5}))^2"/>
    * <img src="https://latex.codecogs.com/gif.latex?b=0.75"/>
    * <img src="https://latex.codecogs.com/gif.latex?K_1=3.5"/>
    * not use query's TF term that in formula (having <img src="https://latex.codecogs.com/gif.latex?K_3"/>'s term), i.e.
        * <img src="https://latex.codecogs.com/gif.latex?\frac{(K_3&plus;1)\times&space;tf_{i,q}}{K_3\times&space;tf_{i,q}}=1" />
* About Rocchio algorithm
    * alpha = 1
    * beta = 0.5
    * gamma = 0.15
    * the number of related documents = 5
    * the number of non-related documents = 1