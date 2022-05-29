#!/usr/bin/env python
# coding: utf-8

# ## Implementation of Monte Carlo Simulation in Python

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set_style("darkgrid")


# In[2]:


# set variables for the Percent to target based on historical results
avg = 1
std_dev = .05
num_reps = 1000
num_simulations = 10000


# In[3]:


# calculating the percent to target
pct_target = np.random.normal(avg, std_dev, num_reps).round(2)


# In[4]:


pct_target[0:10]


# In[5]:


# Another example -- using 'sales target' distribution

sales_target_values = [75_000, 100_000, 200_000, 300_000, 400_000, 500_000]
sales_target_prob = [.3, .3, .2, .1, .05, .05]
sales_target = np.random.choice(sales_target_values, num_reps, p=sales_target_prob)


# In[6]:


sales_target[0:10]


# In[7]:


# creating the dataframe
df = pd.DataFrame(index=range(num_reps), data={'Pct_Target': pct_target,
                                               'Sales_Target': sales_target})
df.head()


# In[8]:


# visualizing the distribution using a histogram 
df['Pct_Target'].plot(kind='hist', title='Historical percent to Target Distribution')
plt.show()


# In[9]:


# Visualizing the sales target distribution
df['Sales_Target'].plot(kind='hist', title='Historical Sales Target Distribution')
plt.show()


# In[10]:



# sales amount
df['Sales'] = df['Pct_Target'] * df['Sales_Target']


# In[11]:


def get_commission_rate(x):
    """ Return the commission rate based on the table:
    0-90% = 2%
    91-99% = 3%
    >= 100 = 5%
    """
    if x <= .90:
        return .02
    if x <= .99:
        return .03
    else:
        return .05


# In[12]:


df['Commission_Rate'] = df['Pct_Target'].apply(get_commission_rate)
df.head()


# In[13]:


#get actual commission amount

df['Commission_Amount'] = df['Commission_Rate'] * df['Sales']
df.head()


# In[14]:


#get sum of each of sales, sales_target and commission amount
df['Sales'].sum(), df['Commission_Amount'].sum(), df['Sales_Target'].sum()


# In[15]:


#key summary statistics
df.describe()


# ### Full Simulation - Combining the results of each round of simulation

# In[16]:


# Initializing a list to keep the results from each round of simulation 
all_stats = []

# Loop through the simulations
for i in range(num_simulations):
    
    # Choosing random inputs for the sales targets and percent to target
    sales_target = np.random.choice(sales_target_values, num_reps, p=sales_target_prob)
    pct_target = np.random.normal(avg, std_dev, num_reps).round(2)
    
    # Dataframe for the inputs and number of reps
    df = pd.DataFrame(index=range(num_reps), data={'Pct_Target': pct_target,
                                                   'Sales_Target': sales_target})
    
    # Back into the sales number using the percent to target rate
    df['Sales'] = df['Pct_Target'] * df['Sales_Target']
    
    # Determining the commission rate and commision amount
    df['Commission_Rate'] = df['Pct_Target'].apply(get_commission_rate)
    df['Commission_Amount'] = df['Commission_Rate'] * df['Sales']
    
    #tracking sales,commission amounts and sales targets over all the simulations
    all_stats.append([df['Sales'].sum().round(0), 
                      df['Commission_Amount'].sum().round(0), 
                      df['Sales_Target'].sum().round(0)])


# In[17]:


results_df = pd.DataFrame.from_records(all_stats, columns=['Sales', 'Commission_Amount', 'Sales_Target'])


# In[18]:


results_df.describe().round().style.format('{:,}')


# In[19]:


results_df['Commission_Amount'].plot(kind='hist', title="Total Commission Amount")
plt.show()


# In[20]:


results_df['Sales'].plot(kind='hist')
plt.show()


# In[ ]:




