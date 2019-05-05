#!/usr/bin/env python
# coding: utf-8

# ## Importa bibliotecas

# In[95]:


import os
import urllib.request
import random
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[96]:


random.seed = 0
np.random.seed = 0


# # Classificação Binária

# ## Download dataset

# In[97]:


data_url = 'http://www.dca.fee.unicamp.br/~lboccato/two_moons.csv'
data_dir = os.path.abspath(os.path.relpath('../data'))
data_path = os.path.join(data_dir, 'two_moons.csv')
image_dir = os.path.abspath(os.path.relpath('../doc/images'))

urllib.request.urlretrieve(data_url, data_path)


# In[98]:


get_ipython().run_cell_magic('bash', '', 'head "../data/two_moons.csv"')


# ## Importa dataset

# In[99]:


dataset = np.loadtxt(data_path, skiprows=1, usecols=(1,2,3), delimiter=',')


# In[100]:


dataset.shape


# In[101]:


X = dataset[:,0:2]
y = dataset[:,2].astype(int)


# In[102]:


mask1 = [i for i, e in enumerate(y) if e]
mask0 = [i for i, e in enumerate(y) if not e]


# In[103]:


plt.plot(X[mask1, 0], X[mask1,1], 'X')
plt.plot(X[mask0, 0], X[mask0, 1], 'o')
plt.xlabel('x0'), plt.ylabel('x1')
plt.title("Distribuição dos dados")
plt.legend(['y=1', 'y=0'])
plt.savefig(os.path.join(image_dir, 'data.png'), bbox_inches='tight')
plt.show()


# ## Discriminante linear de Fischer

# In[104]:


y_hat = lambda w, X: np.dot(w.T, X)


# In[105]:


mean_w = lambda w, X: np.dot(w.T, np.mean(X, 0))


# In[106]:


sd_w = lambda w, X: np.sum(np.square(y_hat(w, X) - mean_w(w, X)))


# In[107]:


s1 =np.sum(np.dot(X[mask1] - np.mean(X[mask1], 0), (X[mask1] - np.mean(X[mask1], 0)).T))


# In[108]:


Sw = np.empty((2,2))
mu1 = np.mean(X[mask1], 0)
mu2 = np.mean(X[mask0], 0)
for i in mask1:
    Sw += np.dot((X[i] - mu1).T,(X[i] - mu1))
for i in mask0:
    Sw += np.dot((X[i] - mu2).T,(X[i] - mu2))
Sw


# In[109]:


Sb = np.dot((np.mean(X[mask1], 0) - np.mean(X[mask0],0)),(np.mean(X[mask1], 0) - np.mean(X[mask0],0)).T)


# In[110]:


J = lambda w: np.dot(np.dot(w.T, Sb), w)/np.dot(np.dot(w.T, Sw), w)


# In[111]:


w = np.dot(np.linalg.inv(Sw),(mu1 - mu2))
w


# In[112]:


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

# In[113]:


proj = lambda X, W: np.dot(X, W)


# In[114]:


out = proj(X, w)
plt.stem(out[mask1], basefmt='.', linefmt='C0.', markerfmt='C0X')
plt.stem(out[mask0], basefmt='.', linefmt='C1.', markerfmt='C1o')
plt.savefig(os.path.join(image_dir, 'stem_proj.png'), bbox_inches='tight')
plt.show()


# In[115]:


plt.hist([out[mask1], out[mask0]], color=['C0', 'C1'])
plt.savefig(os.path.join(image_dir, 'hist.png'), bbox_inches='tight')
plt.show()


# In[116]:


def fischer(X, W, thres):
    out = proj(X, W)
    return np.array([1 if e<thres else 0 for e in out])


# In[117]:


def roc_curve(X, y, w, thres_v):
    roc = []
    for t in thres_v:
        out = fischer(X, w, t)
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
        #print((tp+tn)/(tp+tn+fp+fn))
        roc.append([fpr, tpr])
    return np.array(roc)


# In[118]:


thres_v = [0.001*e for e in range(-30, 30)]
roc = roc_curve(X, y, w, thres_v)
plt.plot(roc[:,0], roc[:,1], 'C0-')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.title("ROC curve")
plt.legend(["tpr/fpr", "tpr=fpr"])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig(os.path.join(image_dir, 'roc.png'), bbox_inches='tight')
plt.show()


# In[ ]:




