#!/usr/bin/env python
# coding: utf-8

# # 1.Importing Libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


# # 2.Dataframe routine data check

# In[2]:


# read csv file downloaded from given website
df =pd.read_csv("application_data.csv")
df.head(2)


# In[3]:


# check shape of the dataset
df.shape


# In[4]:


# check data type of dataset
df.dtypes


# In[5]:


# check null in columns
pd.set_option("max_rows", None)
(df.isnull().sum()*100/len(df)>50).value_counts()


# # 3.Data Analysis for DataFrame(df)

# In[151]:


# checking null value in df in each columns in percentage


# In[7]:


pd.set_option("max_rows", None)
(df.isnull().sum()*100/len(df)).sort_values(ascending=False)


# In[8]:


# Below code gives percentage of null in every column
null_percentage = df.isnull().sum()/df.shape[0]*100

# Below code gives list of columns having more than 47% null
col_to_drop = null_percentage[null_percentage>=47].keys()
print('Many columns having missing value above 47% which is nearly 50% so thought its better to drop that columns')
df = df.drop(col_to_drop, axis=1)
df.shape


# In[9]:


# Below Code to Display Full column range
pd.set_option('display.max_columns', None)
df.head()


# In[10]:


# Selecting columns with less or equal to than 35% null vallues
list(df.columns[(df.isnull().sum()*100/len(df)<=35) & (df.isnull().sum()*100/len(df)>0)])
## lets check those columns for possible imputation


# # 4.2.1. EXT_SOURCE_2 imputation
# 

# In[11]:


# checking value counts
df['EXT_SOURCE_2'].value_counts()


# In[12]:


# datatype of EXT_SOURCE_2 is int and it has numerical values so its continuous variable so we need to check for outliers


# In[13]:


# Below code checking for plot style availability
plt.style.available


# In[14]:


plt.style.use('seaborn')
plt.figure(figsize=[10,5])
sns.boxplot(df['EXT_SOURCE_2'])
plt.show() 


# In[15]:


### hence you can notice in continious variable there is no outlier so it can be treated by either mean or median


# In[16]:


ext_mean=df['EXT_SOURCE_2'].mean()


# In[17]:


# To fill null Value with mean of the column
df['EXT_SOURCE_2'].fillna(ext_mean,inplace=True)


# In[18]:


# Cross checking is column null value is replaced by mean
df['EXT_SOURCE_2'].isnull().sum()


# In[19]:


# plotting box plot after imputation of EXT_SOURCE_2
plt.style.use('seaborn')
plt.figure(figsize=[10,5])
sns.boxplot(df['EXT_SOURCE_2'])
plt.show()


# 
# # 4.2.2. AMT_ANNUITY imputation

# In[20]:


# check for value counts
df['AMT_ANNUITY'].value_counts()


# In[21]:


# Since AMT_ANNUITY is a continuous variable. So checking for outliers
sns.boxplot(df['AMT_ANNUITY'],notch=True)
plt.xticks(fontsize=10)
plt.show()


# In[22]:


# since the column AMT_ANNUITY has very high oultlier so it has to be imputed or treated by median as mean will be interfer by high level or high salary outliers
amt_med=df['AMT_ANNUITY'].median()


# In[23]:


print(f'Replace null value in column AMT_ANNUITY with median value by {amt_med}')


# In[24]:


df.AMT_ANNUITY.fillna(amt_med,inplace=True)
#cross checking if columns is filled by median value
df.AMT_ANNUITY.isnull().sum()


# # 4.2.3. NAME_TYPE_SUITE imputation

# In[25]:


# checking value counts for NAME_TYPE_SUITE column in dataframe(df)
df['NAME_TYPE_SUITE'].value_counts()


# In[26]:


# checking mode of columns and assigning value
name_mode=df['NAME_TYPE_SUITE'].mode()[0]


# In[27]:


print(f'since column NAME_TYPE_SUITE have categorical variable it has to be replace by mode of column that is ={name_mode}')


# In[28]:


# filling missing value in column by name_mode as its categorical column
df.NAME_TYPE_SUITE.fillna(name_mode,inplace=True)
# cross checking is columns is filled by name_mode value
df.NAME_TYPE_SUITE.isnull().sum()


# # 4.2.4. CNT_FAM_MEMBERS imputation

# In[29]:


# checking value counts for column
df.CNT_FAM_MEMBERS.value_counts()


# In[30]:


# column has continuous variable lets check for outliers
sns.boxplot(df['CNT_FAM_MEMBERS'])
plt.show()


# In[31]:


# since column having count for family members have outliers it has to be imputed by median as outlier doesnot interfere in calculation
fam_med =df['CNT_FAM_MEMBERS'].median()
# replacing missing value in column CNT_FAM_MEMBERS
df.CNT_FAM_MEMBERS.fillna(fam_med,inplace=True)
print(f'since column CNT_FAM_MEMBERS have outlier and its continous columns missing value is replaced by median i.e {fam_med}')


# In[32]:


# cross checking if columns is replaced by median 
df.CNT_FAM_MEMBERS.isnull().sum()


# # 4.2.5. AMT_GOODS_PRICE imputation

# In[33]:


# checking for column value count
df.AMT_GOODS_PRICE.value_counts()


# In[34]:


# AMT_GOODS_PRICE is a continuous variable. So checking for outliers
sns.boxplot(df['AMT_GOODS_PRICE'])
plt.show()


# In[35]:


# since we can see in boxplot column has outlier so it has to be imputed by median value
# AMT_GOODS_PRICE is a continuous variable. So checking for outliers
good_med=df.AMT_GOODS_PRICE.median()
# replacing missing value in column 
df.AMT_GOODS_PRICE.fillna(good_med,inplace=True)
print(f'since columns have outlier it has to be imputed by median i.e {good_med}')


# In[36]:


# cross checking if column has replaced missing value with median
df.AMT_GOODS_PRICE.isnull().sum()


# In[37]:


# checking how many columns left with null value
list(df.columns[(df.isnull().sum()*100/len(df)<=35) & (df.isnull().sum()*100/len(df)>0)])


# # 4.2.6 EXT_SOURCE_3 imputation

# In[38]:


# checing value counts
df.EXT_SOURCE_3.value_counts()


# In[39]:


# checking outliers for columns
sns.boxplot(df['EXT_SOURCE_3'])
plt.show()


# In[40]:


# since columns is continuous and have no outlier so we can compute with mean
ext_smean=df['EXT_SOURCE_3'].mean()
# replacing missing value with mean
df.EXT_SOURCE_3.fillna(ext_smean,inplace=True)
# cross checking is columns is replaced by missing value
df.EXT_SOURCE_3.isnull().sum()


# # 4.2.7 Imputing all Credit column

# In[41]:


# since it has outlier it has to be treated by median
sns.boxplot(df.AMT_REQ_CREDIT_BUREAU_YEAR)


# In[42]:


# since all columns below has outliers and overlapp exactly with each other so we can impute all columns by median

data_credit=['AMT_REQ_CREDIT_BUREAU_MON','AMT_REQ_CREDIT_BUREAU_DAY','AMT_REQ_CREDIT_BUREAU_WEEK','AMT_REQ_CREDIT_BUREAU_QRT']
for i in data_credit:
    sns.boxplot(df[i])
    plt.show()


# In[43]:


# checking median for all credit columns
print(df.AMT_REQ_CREDIT_BUREAU_QRT.median())
print(df.AMT_REQ_CREDIT_BUREAU_DAY.median())
print(df.AMT_REQ_CREDIT_BUREAU_WEEK.median())
print(df.AMT_REQ_CREDIT_BUREAU_MON.median())
print(df.AMT_REQ_CREDIT_BUREAU_HOUR.median())
print(df.AMT_REQ_CREDIT_BUREAU_YEAR.median()) 


# In[44]:


zero_med=df.AMT_REQ_CREDIT_BUREAU_QRT.median()
one_med=df.AMT_REQ_CREDIT_BUREAU_YEAR.median()


# In[45]:


# replacing all missing value in column with median
df.AMT_REQ_CREDIT_BUREAU_QRT.fillna(zero_med,inplace=True)
df.AMT_REQ_CREDIT_BUREAU_DAY.fillna(zero_med,inplace=True)
df.AMT_REQ_CREDIT_BUREAU_WEEK.fillna(zero_med,inplace=True)
df.AMT_REQ_CREDIT_BUREAU_MON.fillna(zero_med,inplace=True)
df.AMT_REQ_CREDIT_BUREAU_HOUR.fillna(zero_med,inplace=True)
df.AMT_REQ_CREDIT_BUREAU_YEAR.fillna(one_med,inplace=True)


# In[46]:


# checking number of column with missing value
list(df.columns[(df.isnull().sum()*100/len(df)<=35) & (df.isnull().sum()*100/len(df)>0)])


# # 4.2.8 Imputing Social_circle Columns

# In[47]:


# checking median and mean for all columns
print(df.OBS_30_CNT_SOCIAL_CIRCLE.aggregate(['mean','median']))
print(df.DEF_30_CNT_SOCIAL_CIRCLE.aggregate(['mean','median']))
print(df.OBS_60_CNT_SOCIAL_CIRCLE.aggregate(['mean','median']))
print(df.DEF_60_CNT_SOCIAL_CIRCLE.aggregate(['mean','median']))
data=['OBS_30_CNT_SOCIAL_CIRCLE','DEF_30_CNT_SOCIAL_CIRCLE','OBS_60_CNT_SOCIAL_CIRCLE','DEF_60_CNT_SOCIAL_CIRCLE']


# In[48]:


for i in data:
    sns.boxplot(df[i])
    plt.show()    


# In[49]:


# replacing all missing value with median as it has outlier and continuous variable
obs=df.OBS_30_CNT_SOCIAL_CIRCLE.median()
Def=df.DEF_30_CNT_SOCIAL_CIRCLE.median()
obs6=df.OBS_60_CNT_SOCIAL_CIRCLE.median()
def6=df.DEF_60_CNT_SOCIAL_CIRCLE.median()
df.OBS_30_CNT_SOCIAL_CIRCLE.fillna(obs,inplace=True)
df.DEF_30_CNT_SOCIAL_CIRCLE.fillna(Def,inplace=True)
df.OBS_60_CNT_SOCIAL_CIRCLE.fillna(obs6,inplace=True)
df.DEF_60_CNT_SOCIAL_CIRCLE.fillna(def6,inplace=True)


# In[50]:


df.head(3)


# In[51]:


# reading file for previous data
pdata = pd.read_csv('previous_application.csv',header =0)
pdata.head(2)


# In[52]:


# checking number of column with missing value
list(df.columns[(df.isnull().sum()*100/len(df)<=35) & (df.isnull().sum()*100/len(df)>0)])


# # 4.2.8 OCCUPATION_TYPE Imputation

# In[53]:


# Column Occupation type is categorical column and have missing value upto 30%
#hence we can make variable of 'Missing Occupation ' under specific column
df.OCCUPATION_TYPE.fillna('Missing Occupation',inplace =True)
print(f'Since column has Missing value upto 30% and we cannot drop the column so we can replace it by Missing Occupation')


# In[54]:


# Cross Check Occupation Column if all NAN filled by Missing Occupatiom
df.OCCUPATION_TYPE.isnull().sum()


# # 4.2.9 DAYS_LAST_PHONE_CHANGE Imputation

# In[55]:


df.DAYS_LAST_PHONE_CHANGE.value_counts()


# In[56]:


#Days cant be negative so we can convert column into absolute columns removing negative sign
df['DAYS_LAST_PHONE_CHANGE']=df["DAYS_LAST_PHONE_CHANGE"].abs()


# In[57]:


# boxplot column to check if above columns has any outliers
sns.boxplot(df.DAYS_LAST_PHONE_CHANGE)
plt.show()


# In[58]:


# hence you can check column has outlier so we can impute it by median
phone_med=df.DAYS_LAST_PHONE_CHANGE.median()


# In[59]:


df.DAYS_LAST_PHONE_CHANGE.fillna(phone_med,inplace=True)
print(f'Since Columns has outliers we can impute it by median i.e {phone_med}')


# In[60]:


## checking columns for its datatypes


# In[61]:


df.select_dtypes(include='float').columns


# In[62]:


df.select_dtypes(include='object').columns


# In[63]:


df.select_dtypes(include='int').columns


# In[64]:


# checking dataframe
df.head()


# In[65]:


# Value counts for gender column 
df.CODE_GENDER.value_counts()


# In[66]:


# we can check in Gender columns four value is missing which we can drop as it wont affect our analysis
df =df[~(df['CODE_GENDER']=='XNA')]


# In[67]:


# cross checking if rows with column valueXNA is drop or not
df.CODE_GENDER.value_counts()


# # 4.3 Bining Variable for better analysis

# In[68]:


# lets check quantile for Income for individual and how we can bin for better analysis
df['AMT_INCOME_TOTAL'].quantile([0,0.1,0.3,0.6,0.8,1])


# In[69]:


# Creating A new categorical variable based on income total
df['INCOME_BIN']=pd.qcut(df['AMT_INCOME_TOTAL'],
                                       q=[0,0.1,0.3,0.6,0.8,1],
                                       labels=['VeryLow','Low','Medium','High','VeryHigh'])


# In[70]:


# checking value counts for Birth columns which indicate age in days from day of applications
df.DAYS_BIRTH.value_counts()


# In[71]:


# Applying absolute formula as Age can be negative - sign in columns is just indications
df['DAYS_BIRTH'] =df['DAYS_BIRTH'].abs()


# In[72]:


# lets Create Age group in a year
df["Age"]=(df['DAYS_BIRTH'])//365.25


# In[73]:


df.Age.describe()


# In[74]:


# We can see People who apply for loan are from age 20 to 69 so we can create Age Group for better analysis 
# we can make bin at  5 year interval
df['AGE_BIN']=pd.cut(df['Age'],bins=np.arange(20,71,5))


# In[75]:


df.AGE_BIN.head()


# In[76]:


#lets check for amount credit column
df.AMT_CREDIT.describe()


# # 4.4 ANALYSISING TARGET COLUMNS
# 
# 

# In[77]:


# checking Percentage value of defaulters (1) and non-defaulters
df.TARGET.value_counts(normalize=True)*100


# In[78]:


plt.pie(df['TARGET'].value_counts(normalize=True)*100,labels=['Non Defaulter(Target=0)','Defaulters(Target=1)'],colors=('blue','red'),explode=(0,0.1),autopct='%1.3f%%',textprops={'fontsize':15 })
plt.title('Target Variable -Non Defaulters vs Defaulters',fontdict={'fontsize':20,'fontweight':20})

plt.show()


# #### hence its seem ratio of defaulter to non defaulter is less only 8% are defaulter as per dataset
# 
# #### Lets Minimise Loss of Bank and reduce Percentage of Defaulter by proper analysis of Consumer

# # 4.5 Univariate Analysis

# In[79]:


df.head()


# ### Categorical unordered univariate analysis

# Unordered data do not have the notion of high-low, more-less etc. Example:
# 
# Income,Education,Housing Type of Person
# Marital status 
# Type of Loan
# Occupation & Organization Type of person
# 

# In[80]:


# calculate the percentage of each marital status category. 
df.NAME_FAMILY_STATUS.value_counts(normalize=True)*100


# In[81]:


df.NAME_FAMILY_STATUS.value_counts(normalize=True).plot.bar()
plt.show()
print('Married People are who take more loans')


# In[82]:


# check percentage for gender who take more loan and plotting  bar chart for it
df.CODE_GENDER.value_counts(normalize=True)*100


# In[83]:


df.CODE_GENDER.value_counts(normalize=True).plot.barh()
plt.show()
print('Female seem to take more loan')


# In[84]:


# check Percentage for occupation Type and plot graph for it ,lets check who take more loan
df.OCCUPATION_TYPE.value_counts(normalize=True)*100


# In[85]:


print('hence we can check Occupation are missing and cannot analyis which occupation take more loan ,after missing value labour seem to take more loan')


# In[86]:


df.ORGANIZATION_TYPE.value_counts(normalize=True)*100


# In[152]:


print('People who has business take more loan')


# ## Categorical ordered univariate analysis

# In[88]:


# checking for car percantage if available or not
df.FLAG_OWN_CAR.value_counts(normalize=True)*100


# In[89]:


# checking for realty value
df.FLAG_OWN_REALTY.value_counts(normalize=True)*100


# In[90]:


# checking for educated level of peoplw who take more loan
df.NAME_EDUCATION_TYPE.value_counts(normalize=True)*100


# In[91]:


# plot the pie chart of education categories
df.NAME_EDUCATION_TYPE.value_counts(normalize=True).plot.pie()
plt.show()


# In[92]:


print('Percantage of secondary leverl educated people who take loan are more')


# In[93]:


df.INCOME_BIN.value_counts(normalize=True)


# In[94]:


print('Medium Income People Take more Loan')


# In[95]:


df.AGE_BIN.value_counts(normalize=True)*100


# In[96]:


df.AGE_BIN.value_counts(normalize=True).plot.barh()
plt.show()
print('Age group between 35 to 40 take more loan')


# ### Segmented Univariate analysis

# In[97]:


df0=df[df.TARGET==0]    # Dataframe with all the data related to non-defaulters
df1=df[df.TARGET==1]    # Dataframe with all the data related to defaulters


# In[98]:


# function to count plot for categorical variables
def plotdetail(var):

    plt.style.use('seaborn')
    sns.despine
    fig,(ax1,ax2) = plt.subplots(1,2,figsize=(20,5))
    
    sns.countplot(x=var, data=df0,ax=ax1)
    ax1.set_ylabel('Total Counts')
    ax1.set_title(f'Distribution of {var} for Non-Defaulters',fontsize=15)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=40, ha="right")
    
    # Adding the normalized percentage for easier comparision between defaulter and non-defaulter
    for p in ax1.patches:
        ax1.annotate('{:.1f}%'.format((p.get_height()/len(df0))*100), (p.get_x()+0.1, p.get_height()+50))
        
    sns.countplot(x=var, data=df1,ax=ax2)
    ax2.set_ylabel('Total Counts')
    ax2.set_title(f'Distribution of {var} for Defaulters',fontsize=15)    
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=40, ha="right")
    
    # Adding the normalized percentage for easier comparision between defaulter and non-defaulter
    for p in ax2.patches:
        ax2.annotate('{:.1f}%'.format((p.get_height()/len(df1))*100), (p.get_x()+0.1, p.get_height()+50))
    
    plt.show()


# # Univariant analysis for categorical variable

# In[99]:


# check for defauter and non defaulter for Code Gender
plotdetail('CODE_GENDER')


# Hence we can check Female took more loan and are also more defaulter as compare to male.But rate of male defaulter is more compare to Female

# In[100]:


Default_rate_female =((df1['CODE_GENDER']).value_counts())[0]/((df0['CODE_GENDER']).value_counts())[0]
print(f'{(Default_rate_female)*100}%')


# In[101]:


Default_rate_male =((df1['CODE_GENDER']).value_counts())[1]/((df0['CODE_GENDER']).value_counts())[1]
print(f'{(Default_rate_male)*100}%')


# In[102]:


plotdetail('NAME_INCOME_TYPE')


# we can clearly see working class who owe more loan as compare to other class and they are more defaulter and should be studied more before loan processing
# 
# 
# student,businessman  are safe for loan process
# 

# In[103]:


plotdetail('FLAG_OWN_CAR')


# People without car buy more loan and rate of defaulting is also more than people who has car
# 
# so People without car background should be studied in more detail comparing people with car for loan Process.
# 

# In[104]:


plotdetail('NAME_FAMILY_STATUS')


# Married People take more and defaulter is also more.
# 
# Defaulters Rate of Single and civil marriage people are high as they contribute low in taking loan and defaulting is high
# 

# In[105]:


plotdetail('NAME_HOUSING_TYPE')


# From graph we can see Person with house tend to appy for more loan and they default also as compare to other
# 
# Risk is with Person staying with parent & in rent  are more likely to default

# # Univariant categorical ordered analysis

# In[106]:


plotdetail('AGE_BIN')


# We see that (25,30] age group tend to default more often. So they are the riskiest people to loan to.
# With increasing age group, people tend to default less starting from the age 25. One of the reasons could be they get employed around that age and with increasing age, their salary also increases

# In[107]:


plotdetail('INCOME_BIN')


# Person with very high salary tend to default lower.
# 
# But with very low and low salary they tend to default more 

# In[108]:


plotdetail('NAME_EDUCATION_TYPE')


# seconday education person tend to default more as compared to other.Its riskier to clear loan who are less educated.
# 
# 
# Higher education are safer as comapared to other education level person as they earn more due to their education background

# In[109]:


plotdetail('REGION_RATING_CLIENT')


# More People from second tier city tend to apply for loan and they seem to be more defaulter also.
# 
# People from third tier they tend to contribute less and default ration is higher as compared to othere region

# In[110]:


plotdetail('FLAG_OWN_REALTY')


# Person who own Realty are more secure than who dont have realty

# # Univariate continuous variable analysis

# In[111]:


def plotcon(var):

    plt.style.use('seaborn')
    sns.despine
    fig,(ax1,ax2) = plt.subplots(1,2,figsize=(15,5))
    
    sns.distplot(a=df0[var],ax=ax1)

    ax1.set_title(f'Distribution of {var} for Non-Defaulters',fontsize=20)
            
    sns.distplot(a=df1[var],ax=ax2)
    ax2.set_title(f'Distribution of {var} for Defaulters',fontsize=15)    
        
    plt.show()


# In[112]:


plotcon('AMT_GOODS_PRICE')


# From above Graph its very difficult to assume but people who want to default go with lower range good product

# In[113]:


plotcon('AMT_INCOME_TOTAL')


# Hence we can see lower Income tend to buy more loan and are more prone to default

# In[114]:


plotcon('AMT_CREDIT')


# Hence its same as AMT_GOOD_PRICE columns as credit will be equal to loan bout on Good price

# In[115]:


plotcon('DAYS_EMPLOYED')


# Recently joined Employe are default more and its riskier for bank to process for loan 
# 
# Its due to on going expense they wont be able to pay installment on time

# In[116]:


plotcon('CNT_FAM_MEMBERS')


# Family with 2 people tend to apply for loan more, but family with 3 people tend to default more and bank is at higher risk to process for loan before checking annual income and other analysis

# In[117]:


#Lets check defaulter  with respect to Income Amount 
sns.countplot(data=df,y='NAME_EDUCATION_TYPE',hue='TARGET')
plt.show()


# Lower and Seconday Education with Salary also seem to be defaulter .Bank is at high risk to Process loan for lower educated Person.

# In[118]:


sns.scatterplot(data=df,x='OCCUPATION_TYPE',y='AMT_INCOME_TOTAL',hue='TARGET')
plt.xticks(rotation=90)
plt.show()


# Its difficult to analysis on base of above scatter plot 

# In[ ]:





# In[119]:


#lets check list of columns
column_list=[]
for i in df.columns:
    column_list.append(i)
print(column_list) 


# In[120]:


#Its very difficult to find correlation among all the columns so lets select few important columns  can be analyis for correlatiom
corr=df[['AMT_GOODS_PRICE','AMT_CREDIT','AMT_INCOME_TOTAL','DAYS_EMPLOYED','DAYS_BIRTH','CNT_FAM_MEMBERS','REGION_RATING_CLIENT_W_CITY','OBS_30_CNT_SOCIAL_CIRCLE','DEF_30_CNT_SOCIAL_CIRCLE','TARGET']].corr()
sns.heatmap(corr,cmap='Reds',annot=True)


# From above heatmap we can check there  correlation with Target Variable

# In[121]:


df.groupby(by='TARGET')['DEF_30_CNT_SOCIAL_CIRCLE'].value_counts().plot.barh()


# Defaulting in social circle is lower 

# # Bivariate Analysis of numerical variables

# In[122]:


def plotnum(var,var1):

    plt.style.use('seaborn')
    sns.despine
    fig,(ax1,ax2) = plt.subplots(1,2,figsize=(20,10))
    
    sns.scatterplot(x=var, y=var1,data=df0,ax=ax1)
    ax1.set_xlabel(var)    
    ax1.set_ylabel(var1)
    ax1.set_title(f'{var} vs {var1} for Non-Defaulters',fontsize=15)
    
    sns.scatterplot(x=var, y=var1,data=df1,ax=ax2)
    ax2.set_xlabel(var)    
    ax2.set_ylabel(var1)
    ax2.set_title(f'{var} vs {var1} for Defaulters',fontsize=15)
        
    plt.show()


# In[123]:


plotnum('AMT_INCOME_TOTAL','CNT_FAM_MEMBERS')


# Lower income with family member at higher risk for defaulting

# In[124]:


plotnum('INCOME_BIN','DEF_30_CNT_SOCIAL_CIRCLE')


# there is no exact correlation between Income Group  and Social circle. Defaulter at lower income can risk bank at loss

# In[125]:


plotnum('AMT_INCOME_TOTAL','REGION_RATING_CLIENT_W_CITY')


# Its riskier for bank to process loan for low salaried or Income group from any region

# In[126]:


plotnum('AMT_INCOME_TOTAL','OCCUPATION_TYPE')


# # Data Analysis For Previous Application Data

# ## Checking for Previous application Data frame
# 

# In[127]:


pdf = pd.read_csv('previous_application.csv')


# In[128]:


pdf.head(2)


# In[129]:


pdf.shape


# In[130]:


# Removing all the columns with more than 50% of null values
pdf = pdf.loc[:,pdf.isnull().mean()<=0.5]
pdf.shape


# In[ ]:





# ### Univariate analysis

# In[131]:


# function to count plot for categorical variables
def plotpdf(var):

    plt.style.use('seaborn')
    sns.despine
    fig,ax = plt.subplots(1,1,figsize=(15,5))
    
    sns.countplot(x=var, data=pdf,ax=ax,hue='NAME_CONTRACT_STATUS')
    ax.set_ylabel('Total Counts')
    ax.set_title(f'Distribution of {var}',fontsize=15)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
    
    plt.show()


# In[132]:


plotpdf('NAME_CONTRACT_TYPE')


# Cash Loan rejection rate is much higher than any other type of loan
# 
# Consumer Loan approval rate is higher than any other type

# In[133]:


plotpdf('NAME_PAYMENT_TYPE')


# From above given data analysis it seem consumer repay loan in form of cash rather than cashless way

# In[134]:


plotpdf('NAME_CLIENT_TYPE')


# Consumer who are repeatedly taking loan are approved very easily, who has any default before may get rejected due to any of reason delay of repayment,lost of job or many other factors

# In[135]:


pdf.NAME_GOODS_CATEGORY.value_counts()


# In[ ]:





# Value is missing for which loan is mostly approved difficult analysis on above plot

# # Correlation in previous application dataset

# In[136]:


corrp=pdf.corr()
corr_df = corrp.where(np.triu(np.ones(corrp.shape),k=1).astype(np.bool)).unstack().reset_index()
corr_df.columns=['Column1','Column2','Correlation']
corr_df.dropna(subset=['Correlation'],inplace=True)
corr_df['Abs_Correlation']=corr_df['Correlation'].abs()
corr_df = corr_df.sort_values(by=['Abs_Correlation'], ascending=False)
corr_df.head(10)


# # PAIRPLOT FOR TOP NUMBERICAL VARIABLE

# In[137]:


plt.figure(figsize=[15,5])
sns.pairplot(pdf[['AMT_ANNUITY','AMT_APPLICATION','AMT_CREDIT','AMT_GOODS_PRICE']],diag_kind = 'kde',plot_kws = {'alpha': 0.5, 's': 70, 'edgecolor': 'k'},size = 5)
plt.show()


# Annuity has positive influence over Good price,Amount of Credit, Amount of application
# 
# Amount of Credit asked by client is directly influence by Good price and application Amount

# In[138]:


#by variant analysis function
def plot_var(var, var1):

    plt.style.use('seaborn')
    sns.despine
    fig,ax = plt.subplots(1,1,figsize=(10,5))
    
    sns.boxplot(x=var,y = var1, data=pdf)
    ax.set_ylabel(f'{var1}')
    ax.set_xlabel(f'{var}')

    ax.set_title(f'{var} Vs {var1}',fontsize=15)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
     
    plt.show()


# In[139]:


plot_var('NAME_CONTRACT_STATUS', 'AMT_CREDIT')


# Low amount of credit get cancelled by bank may be due to lower profit and higher work

# In[140]:


plot_var('NAME_CONTRACT_STATUS', 'AMT_ANNUITY')


# Loan getting Cancelled for low annuity may be having other issue like client want to cancel the loan
# 
# Higher annuity are refused more often

# # Merging New application dataset and Previous application dataset

# In[141]:


#lets merge file for analysis
mdf = pd.merge(df, pdf, how='left', on=['SK_ID_CURR'])


# In[142]:


mdf.head()


# In[143]:


mdf.shape


# In[144]:


pdf.drop(['SK_ID_CURR','WEEKDAY_APPR_PROCESS_START', 'HOUR_APPR_PROCESS_START', 'FLAG_LAST_APPL_PER_CONTRACT','NFLAG_LAST_APPL_IN_DAY'],axis=1,inplace=True)


# In[145]:


#checking and removing missing value from cash loan purpose
mdf.NAME_CASH_LOAN_PURPOSE.value_counts()
mdf=mdf[~(mdf['NAME_CASH_LOAN_PURPOSE']=='XAP')]
mdf=mdf[~(mdf['NAME_CASH_LOAN_PURPOSE']=='XNA')]


# In[146]:


mdf.NAME_CASH_LOAN_PURPOSE.value_counts()


# In[147]:


#lets check which Purpose Loan is most approved
sns.set_style('whitegrid')
sns.set_context('talk')

plt.figure(figsize=(15,30))
plt.rcParams["axes.labelsize"] = 20
plt.rcParams['axes.titlesize'] = 22
plt.rcParams['axes.titlepad'] = 30
plt.xticks(rotation=90)
plt.xscale('log')
plt.title('Distribution of contract status with purposes')
ax = sns.countplot(data = mdf, y= 'NAME_CASH_LOAN_PURPOSE', 
                   order=mdf['NAME_CASH_LOAN_PURPOSE'].value_counts().index,hue = 'NAME_CONTRACT_STATUS',palette='magma')  


# Education Loan are highly approved and has very low chance of deafulting.
# 
# If Household Income and Job Category is checked very well
# Repair purpose is highly rejected
# Building House,Buying Car,Loan for third oerson is also rejected.
# But with proper background study we can analysis whether to approve loan or not

# In[148]:



sns.set_style('whitegrid')
sns.set_context('talk')

plt.figure(figsize=(15,30))
plt.rcParams["axes.labelsize"] = 30
plt.rcParams['axes.titlesize'] = 30
plt.rcParams['axes.titlepad'] = 30
plt.xticks(rotation=90)
plt.xscale('log')
plt.title('Distribution of contract status with purposes')
ax = sns.countplot(data = mdf, y= 'NAME_CASH_LOAN_PURPOSE', 
                   order=mdf['NAME_CASH_LOAN_PURPOSE'].value_counts().index,hue = 'NAME_CONTRACT_STATUS',palette='magma') 


# Loan for Repair have highest chance of defaulting
# 
# Hence we can see in above plot for all purpose their is chance of defaulting.
# 
# I assume bank should never approve loan without checking Income,Age,Working Status,Realty,Car,Family Background
# If Realty or some assets are available bank can recover money easily
# 
#  

# In[149]:


plt.figure(figsize=(16,12))
plt.xticks(rotation=90)
sns.barplot(data =mdf, y='AMT_CREDIT_y',hue='TARGET',x='NAME_HOUSING_TYPE')
plt.title('Housing type vs Credit previous time')
plt.show()


# From above Boxplot who stay in co-op appartment seem to be more defaulters
# Defaulting have almost same chance of defaulting

# In[ ]:





# In[ ]:





# # Conclusion

# ### 1. Bank should focus on high and medium income group who are well qualifed ,as it seem lower educated People have very difficult to replay the loan
# 
# ### 2.Consumer who have high family member and Low income also tend to be defaulter
# 
# ### 3.Consumer who have their own house and stay with parent have less chance of defaulting
# 
# ### 4. Loan for repair work seem to have difficulty to repay the loan
# 
# ### 5.Consumer with car & realty have everyless chance of defaulting, Bank can even recover money if Consumer is not able to repay the loan
# 
# ### 6.Business and pensioner are having very low chance of defaulting than working class
# 
# ### 7.Practically if consumer is genuine Income play very vital role for loan repayment.Any Occupation class who have less Income cant pay Loan on time.If consumer is doing job salary slip of 3-6 month should be check if they getting salary on time or not, if business person  ITR should be check for yearly income of person.
# 
# ### 8.If Loan is provided againt any security is very safer.
# 
# 

# In[ ]:





# In[150]:


pip install nbconvert


# In[154]:


conda install nbconvert


# In[ ]:




