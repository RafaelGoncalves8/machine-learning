#!/usr/bin/env python
# coding: utf-8

# ## Importa bibliotecas

# In[42]:


import os
import urllib.request
import random
import numpy as np
import matplotlib.pyplot as plt
import csv
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


random.seed = 0
np.random.seed = 0


# # Classificação Binária

# ## Download dataset

# In[3]:


data_url = 'http://www.dca.fee.unicamp.br/~lboccato/two_moons.csv'
data_dir = os.path.abspath(os.path.relpath('../data'))
data_path = os.path.join(data_dir, 'two_moons.csv')
image_dir = os.path.abspath(os.path.relpath('../doc/images'))

urllib.request.urlretrieve(data_url, data_path)


# In[4]:


get_ipython().run_cell_magic('bash', '', 'head "../data/two_moons.csv"')


# ## Importa dataset

# In[5]:


dataset = np.loadtxt(data_path, skiprows=1, usecols=(1,2,3), delimiter=',')


# In[6]:


dataset.shape


# In[7]:


X = dataset[:,0:2]
y = dataset[:,2].astype(int)


# In[8]:


Phi = np.column_stack((np.ones(X.shape[0]), X))


# In[9]:


mask1 = [i for i, e in enumerate(y) if e]
mask0 = [i for i, e in enumerate(y) if not e]


# In[10]:


plt.plot(X[mask1, 0], X[mask1,1], 'X')
plt.plot(X[mask0, 0], X[mask0, 1], 'o')
plt.xlabel('x0'), plt.ylabel('x1')
plt.title("Distribuição dos dados")
plt.legend(['y=1', 'y=0'])
plt.savefig(os.path.join(image_dir, 'data.png'), bbox_inches='tight')
plt.show()


# ## Discriminante linear de Fischer (LDA)

# In[11]:


y_hat = lambda w, X: np.dot(X, w)


# In[12]:


Sw = np.empty((2,2))
mu1 = np.mean(X[mask1], 0)
mu2 = np.mean(X[mask0], 0)
for i in mask1:
    Sw += np.dot((X[i] - mu1).T,(X[i] - mu1))
for i in mask0:
    Sw += np.dot((X[i] - mu2).T,(X[i] - mu2))
Sw


# In[13]:


Sb = np.dot((np.mean(X[mask0], 0) - np.mean(X[mask1],0)),(np.mean(X[mask0], 0) - np.mean(X[mask1],0)).T)


# In[14]:


J = lambda w: np.dot(np.dot(w.T, Sb), w)/np.dot(np.dot(w.T, Sw), w)


# In[15]:


w = np.dot(np.linalg.inv(Sw),(mu1 - mu2))
w


# In[16]:


plt.plot(X[mask1, 0], X[mask1,1], 'X', zorder=1)
plt.plot(X[mask0, 0], X[mask0, 1], 'o', zorder=2)


origin = [0], [0] # origin point
plt.xlabel('x0'), plt.ylabel('x1')
plt.title("Direção de projeção")
plt.quiver(*origin, w[0], w[1], width=0.006, color='black', zorder=3)

# w0 = 0
# x = np.linspace(-1.5, 2.5, 1000)
# gx = -(w[0]/w[1])*x - (w[0]/w[1])*w0
# plt.plot(x, gx, 'r', zorder=4)

plt.legend(['y=1', 'y=0', 'w', 'thres = 0'])
plt.savefig(os.path.join(image_dir, 'proj.png'), bbox_inches='tight')
plt.show()


# ### Projeção em w

# In[17]:


lda = lambda X, w: -np.dot(X, w)


# In[18]:


out = lda(X, w)
plt.stem(out[mask1], basefmt='.', linefmt='C0.', markerfmt='C0X')
plt.stem(out[mask0], basefmt='.', linefmt='C1.', markerfmt='C1o')
plt.savefig(os.path.join(image_dir, 'stem_proj.png'), bbox_inches='tight')
plt.show()


# In[19]:


plt.hist([out[mask1], out[mask0]], color=['C0', 'C1'])
plt.savefig(os.path.join(image_dir, 'hist.png'), bbox_inches='tight')
plt.show()


# ### Curva ROC

# In[20]:


def decision(X, w, thres, model):
    out = model(X, w)
    return np.array([1 if e > thres else 0 for e in out])


# In[21]:


def roc_curve(X, y, w, thres_v, model):
    roc = []
    f1_v = []
    for t in thres_v:
        out = decision(X, w, t, model)
        tp = fp = tn = fn = 0
        for i, e in enumerate(out):
            if e:
                if y[i]:
                    tp += 1
                else:
                    fp += 1
            else:
                if y[i]:
                    fn += 1
                else:
                    tn += 1
        tpr = tp/(tp+fn)
        fpr = fp/(fp+tn)
        f1 = 2*tp/(2*tp + fn + fp)
        f1_v.append([t, f1])
        roc.append([fpr, tpr])
    return np.array(roc), np.array(f1_v)


# In[22]:


thres_v = [0.001*e for e in range(-30, 31)]
roc, f1 = roc_curve(X, y, w, thres_v, lda)
plt.plot(roc[:,0], roc[:,1], 'C0-')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.title("ROC curve")
plt.legend(["ROC", "tpr=fpr"])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig(os.path.join(image_dir, 'roc.png'), bbox_inches='tight')
plt.show()


# ### F1 score

# In[23]:


plt.plot(f1[:,0], f1[:,1], 'C0-')
plt.title("F1 score x threshold value")
plt.ylabel('F1 Score')
plt.xlabel('Threshold')
plt.savefig(os.path.join(image_dir, 'f1.png'), bbox_inches='tight')
plt.show()


# ## Regressão Logística

# In[24]:


g = lambda z: 1/(1 + np.exp(-z))


# In[25]:


lr = lambda Phi, w: g(np.dot(Phi, w))


# In[26]:


mmq = lambda Phi, y: np.dot(np.dot(np.linalg.inv(np.dot(Phi.T, Phi)), Phi.T), y)


# In[27]:


w = mmq(Phi, y)


# In[32]:


thres_v = [0.001*e for e in range(300,701, 5)]
roc, f1 = roc_curve(Phi, y, w, thres_v, lr)
plt.plot(roc[:,0], roc[:,1], 'C0-')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.title("ROC curve")
plt.legend(["ROC", "tpr=fpr"])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig(os.path.join(image_dir, 'roc_lr.png'), bbox_inches='tight')
plt.show()


# In[33]:


plt.plot(f1[:,0], f1[:,1], 'C0-')
plt.title("F1 score x threshold value")
plt.ylabel('F1 Score')
plt.xlabel('Threshold')
plt.savefig(os.path.join(image_dir, 'f1_lr.png'), bbox_inches='tight')
plt.show()


# # Classificação multi-classe

# ## Download dataset

# In[34]:


data_url = 'http://www.dca.fee.unicamp.br/~lboccato/dataset_vehicle.csv'
data_path = os.path.join(data_dir, 'dataset_vehicle.csv')

urllib.request.urlretrieve(data_url, data_path)


# In[36]:


get_ipython().run_cell_magic('bash', '', 'head "../data/dataset_vehicle.csv"')


# ## Importa dataset

# In[40]:


X = np.loadtxt(data_path, skiprows=1, usecols=range(18), delimiter=',')
X.shape


# In[52]:


y = []
with open(data_path) as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        y.append(row['Class'])   
len(y)


# In[57]:


classes = list(set(y))
classes


# In[58]:


Y = np.zeros((len(y), len(classes)))
for i, e in enumerate(y):
    if e == classes[0]:
        Y[i,0] = 1
    elif e == classes[1]:
        Y[i,1] = 1
    elif e == classes[2]:
        Y[i,2] = 1
    else:
        Y[i,3] = 1
Y.shape


# In[61]:


holdout_n = int(0.3*len(y))


# In[66]:


X_test = X[:holdout_n, :]
X_train = X[holdout_n:,:]
Y_test = Y[:holdout_n, :]
Y_train = Y[holdout_n:,:]
X_train.shape, X_test.shape, Y_train.shape, Y_test.shape


# ## Regressão Logística

# In[ ]:




