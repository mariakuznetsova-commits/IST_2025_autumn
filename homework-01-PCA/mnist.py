#!/usr/bin/env python
# coding: utf-8
# # Рабочая тетрадь - распознавание рукописных цифр
# In[1]:
import numpy as np
from sklearn.datasets import fetch_openml

def load_mnist():
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X = mnist['data'].astype('float32') / 255.0  # (70000, 784)
    y = mnist['target'].astype('int64')

    # 50k train, 10k val, 10k test
    X_train, y_train = X[:50000], y[:50000]
    X_val,   y_val   = X[50000:60000], y[50000:60000]
    X_test,  y_test  = X[60000:], y[60000:]

    train = (X_train, y_train)
    validation = (X_val, y_val)
    test = (X_test, y_test)
    return train, validation, test
'''#import mnist
# #!pip install scikit-learn
# In[2]:
#train, validation, test = mnist.load_mnist()
# In[3]:
x = train[0].reshape(-1, 28, 28)
y = train[1]

x_validation = validation[0].reshape(-1, 28, 28)
y_validation = validation[1]
# In[21]:
y
# In[22]:
x.shape
# In[14]:
import matplotlib.pyplot as plt
# In[23]:
plt.imshow(x[0], cmap="gray")
plt.title(y[0])
# In[45]:
fig, axs = plt.subplots(5,5, figsize=(15,15))
x_subset = x[y==4]
for i in range(5):
    for j in range(5):
        axs[i,j].imshow(x_subset[5*i + j][::2, ::2], cmap="gray")
# In[29]:
from sklearn.neighbors import KNeighborsClassifier as kNN
# In[30]:
clf = kNN()
# In[42]:
clf.fit(x.reshape(-1, 784), y)
# In[43]:
get_ipython().run_cell_magic('time', '', 'y_pred = clf.predict(x_validation.reshape(-1, 784))\n')
# In[32]:
x_validation.shape
# In[35]:
y_pred
# In[36]:
y_validation
# In[41]:
import numpy as np
np.sum(y_pred != y_validation).item() / len(y_validation)
# ## Ресайзим картинки
# In[47]:
clf2 = kNN()
clf2.fit(x[: ,::2,::2].reshape(-1, 14*14), y)
# In[48]:
get_ipython().run_cell_magic('time', '', 'y_pred2 = clf2.predict(x_validation[: ,::2,::2].reshape(-1, 14*14))\n')
# In[49]:
np.sum(y_pred2 != y_validation).item() / len(y_validation)
# ## Уменьшение размерности с помощью PCA
# In[50]:
from sklearn.decomposition import PCA 
pca = PCA(n_components=16)
# In[51]:
pca.fit(x.reshape(-1, 784))
# In[54]:
x_transformed = pca.transform(x.reshape(-1, 784))
x_validation_transformed = pca.transform(x_validation.reshape(-1, 784))
# In[55]:
x_transformed.shape, x_validation_transformed.shape
# In[56]:
clf3 = kNN()
clf3.fit(x_transformed, y)
# In[57]:
get_ipython().run_cell_magic('time', '', 'y_pred3 = clf3.predict(x_validation_transformed)\n')
# In[58]:
np.sum(y_pred3 != y_validation).item() / len(y_validation)
# ## Случайный алгоритм
# In[72]:
y_pred4 = np.random.randint(low=0, high=10, size=len(y_validation))
# In[73]:
y_pred4
# In[76]:
print("доля правильных ответов (accuracy):", np.sum(y_pred4 == y_validation).item() / len(y_validation))
# # Визуализация с помощью t-SNE
# In[4]:
from sklearn.manifold import TSNE
# In[5]:
tsne = TSNE(n_components=2)
# In[7]:
x_tsne = tsne.fit_transform(x[:2000].reshape(-1, 784))
# In[8]:
x_tsne.shape
# In[15]:
import matplotlib.pyplot as plt
plt.scatter(x_tsne[:, 0], x_tsne[:, 1], c=y[:2000], cmap="tab10")
plt.colorbar()
# In[ ]:

'''


