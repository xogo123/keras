
# coding: utf-8

# In[1]:


from __init__ import *
from preprocess import *
from model import *


# In[ ]:


### arguments setting
start_t = time.time()
args = get_args()
model_type = 'CNN'
path_model = get_path_new_model(model_type)
#path_model = './model/CNN_1.h5'


tensorflow_init()


# In[ ]:


### argument and data view


# In[1]:


### preprocess


# In[2]:


### training


# In[3]:


### testing


# In[ ]:


print ('all process cost {} seconds'.format(time.time() - start_t))
print ('all done')

