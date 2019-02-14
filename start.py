
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import cvxopt
import cvxopt.solvers
from collections import Counter
from itertools import combinations_with_replacement
from time import time


# **Reading files**

# In[ ]:


xtr0 = pd.read_csv("Data/Xtr0.csv", " ", header=0)
xtr1 = pd.read_csv("Data/Xtr1.csv", " ", header=0)
xtr2 = pd.read_csv("Data/Xtr2.csv", " ", header=0)
xtrain_temp = np.append(np.append(xtr0, xtr1), xtr2)
xtrain = np.array(xtrain_temp)

xte0 = pd.read_csv("Data/Xte0.csv", " ", header=0)
xte1 = pd.read_csv("Data/Xte1.csv", " ", header=0)
xte2 = pd.read_csv("Data/Xte2.csv", " ", header=0)
xtest_temp = np.append(np.append(xte0, xte1), xte2)
xtest = np.array(xtest_temp)

ytr0 = pd.read_csv("Data/Ytr0.csv", index_col=0, header=0)
ytr1 = pd.read_csv("Data/Ytr1.csv", index_col=0, header=0)
ytr2 = pd.read_csv("Data/Ytr2.csv", index_col=0, header=0)
ytrain_temp = np.append(np.append(ytr0, ytr1), ytr2)
ytrain = np.array(ytrain_temp)
ytrain[ytrain[:] == 0] = -1


# **Preparing features: kmers**

# In[ ]:


def create_kmers(x, y, test, k):
    subseq = create_subsequences(k)
    index = np.arange(0, len(subseq))
    features_train = prepare_data(x, subseq, index, k)

    if test.size != 0:
        features_test = prepare_data(test,subseq, index, k)
        return features_train , features_test
    else:
        return features_train    

def create_subsequences(length):
    p = ['A','C','G','T','C','G','T','A','G','T','A','C','T','A','C','G', 'A','C','G','T']
    subseq = []
    for i in combinations_with_replacement(p, length):
        subseq.append(list(i))
    subseq = np.asarray(subseq)    
    subseq= np.unique(subseq, axis = 0) 
    subseq =["".join(j) for j in subseq[:,:].astype(str)]
    
    return subseq

def prepare_data(x, subsequence, index, k):
    features = np.zeros((len(x), len(subsequence)))   #To store the occurence of each string
    for i in range(0,len(x)):
        s = x[i]
        c = [ s[j:j+k] for j in range(len(s)-k+1) ]
        counter = Counter(c)
        j=0
        for m in subsequence:
            features[i][j] = counter[m]
            j=j+1

    features_train = features[:,index]
    features_train = features_train / np.max(np.abs(features_train),axis=0)
    return features_train


# **Different non linear kernels**

# In[ ]:


def polynomial_kernel(x, y, p = 3):
    return (1 + np.dot(x, y)) ** p

def rbf_kernel(x, y, sigma = 3):
    return np.exp(-np.linalg.norm(x-y)**2 / (2 * (sigma ** 2)))

def linear_kernel(x, y):
    return np.inner(x, y)


# In[ ]:


class SVM(object):
    def __init__(self, kernel = polynomial_kernel, C = 1):
        self.kernel = kernel
        self.C = C
            
    def fit_svm(self, x, y):
        num_obs, num_features = x.shape
        #Gram matrix
        Gram = np.zeros((num_obs, num_obs))
        print("Computing Gram matrix:")
        for i in range(num_obs):
            if (i%100 == 0):
               print(i, "/", num_obs)
            for j in range(num_obs):
                Gram[i,j] = self.kernel(x[i], x[j])  
                
        #Components for quadratic program problem        
        P = cvxopt.matrix(np.outer(y,y) * Gram)
        q = cvxopt.matrix(-np.ones((num_obs, 1)))
        A = cvxopt.matrix(y, (1, num_obs), 'd')
        b = cvxopt.matrix(np.zeros(1))
        diag = np.diag(np.ones(num_obs) * -1)
        identity = np.identity(num_obs)
        G = cvxopt.matrix(np.vstack((diag, identity)))
        h = cvxopt.matrix(np.hstack((np.zeros(num_obs), np.ones(num_obs) * self.C)))
        
        #Solving quadratic progam problem
        sol = cvxopt.solvers.qp(P, q, G, h, A, b)
        alphas = np.ravel(sol['x'])
        
        #Support vectors have non zero lagrange multipliers, cut off at 1e-6
        sup_vec = alphas > 1e-6
        ind = np.arange(len(alphas))[sup_vec]        
        
        #Creating support vectors
        self.alphas = alphas[sup_vec]
        self.sup_vec = x[sup_vec]
        self.sup_vec_y = y[sup_vec]
        
        #Fitting support vectors with the intercept
        self.b = 0
        for i in range(len(self.alphas)):
            self.b += self.sup_vec_y[i]
            self.b -= np.sum(self.alphas * self.sup_vec_y * Gram[ind[i],sup_vec])
        self.b /= len(self.alphas)
        print(self.b)
        
        #Weight for non linear kernel(polynomial or rbf)
        self.w = None  
            
    #Predict the sign
    def predict(self, X):
        y_pred = np.zeros(len(X))
        for i in range(len(X)):
            s = 0
            for alphas, sup_vec_y, sup_vec in zip(self.alphas, self.sup_vec_y, self.sup_vec):
                s += alphas * sup_vec_y * self.kernel(X[i], sup_vec)
            y_pred[i] = s
        return np.sign(y_pred + self.b)


# In[ ]:


#Creating features
x_trainf, x_testf = create_kmers(xtrain, ytrain, xtest, k=5)
#Initialization
svm = SVM(polynomial_kernel, 0.1)
#SVM Fitting
start = time()
svm.fit_svm(x_trainf, ytrain)
#Prediction
print("Predicting:")
prediction = svm.predict(x_testf)
end = time()
print("Training and prediction time is:", round(end - start, 2), "s")


# **Saving Prediction to file**

# In[ ]:


col2 = prediction
col2[col2 == -1] = 0
col2 = col2.astype(int)
col1 = range(0, col2.shape[0])
predictions = pd.DataFrame({'Id': col1, 'Bound': col2})
predictions.to_csv('Yte.csv', sep=',', encoding='utf-8',index=False)
print("Predictions are saved to Yte.csv")

