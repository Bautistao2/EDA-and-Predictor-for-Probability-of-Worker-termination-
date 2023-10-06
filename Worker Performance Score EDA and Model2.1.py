#!/usr/bin/env python
# coding: utf-8

# ## Importing Packages

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LinearRegression
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from simple_colors import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from plotly.offline import init_notebook_mode
init_notebook_mode(connected=True)
from plotnine import *


# ## Load and read data

# In[4]:


# loading the dataset to pandas dataframe
df = pd.read_csv("hrdata.csv",
parse_dates=['DOB', 'DateofHire', 'DateofTermination', 'LastPerformanceReview_Date'])
df.head()


# In[5]:


df.tail()


# In[6]:


#Dataset Size
df.shape


# In[7]:


print("Dataset has {} data  with {} variables each.".format(*df.shape))


# # Explotory Data Analysis (EDA) & Visualization

# In[8]:


df.reindex()


# In[9]:


# statistical measure of dataset
df.describe()


# In[10]:


#Get information about DataFrame
df.info()


# **Commits**
# *We only have NULL values in "DateofTermination" & "ManagerID" columns

# In[11]:


#check missing values
df.isnull().sum()


# In[12]:


df.DateofTermination.value_counts()


# In[13]:


df.fillna("0",inplace=True)
df


# In[14]:


df.isnull().sum()


# In[15]:


df.drop_duplicates(inplace=True)
df


# **Commits**
# *The data types are appropriate.*
# *The dataset covers data on 311 employees.*
# *104 out of 311 employees no longer work for the company.*
# *Missing values in the DateofTermination column mean that 207 persons are still employed*

# In[16]:


#First, we will check and fix the DOB dates of the original %m/%d/%y format, which Pandas might not convert correctly. In other words, some years like 71, 72 etc. can be converted to 2071, 2072.
df.query('@df.DOB.dt.year > 2000').DOB.dt.year.sort_values().unique()


# In[17]:


df['DOB'] = np.where(df['DOB'].dt.year >= df['DateofHire'].dt.year, df['DOB'] - pd.offsets.DateOffset(years=100), df['DOB'])


# In[18]:


df.select_dtypes(include='datetime').describe()


# **Commits**
# The company was founded in 2006 (the minimum DateofHire is 2006-01-09)
# The time period covered by the dataset includes 2019 (the maximum date value in the dataset is 2019-02-28)

# In[19]:


for column in df.select_dtypes(include='O').columns:
    print(blue(f'{column}', 'bold')) 
    print(black('Number of unique values :', 'underlined'), df[column].nunique())
    if column == 'Employee_Name':
        print(df[column].unique()[:30])
    else:
        print(df[column].unique())
    print()


# **Commits**
# 
# 
# Extra white spaces found in the columns Employee_Name (as well as the lack of white spaces), Department
# 
# Different format of names in the Employee_Name and ManagerName columns. It would be useful to bring them to the same format in order to check which of the managers is an employee of the company
# 
# The column HispanicLatino partially duplicates the column RaceDesc and has different formats for recording the same values: 'No', 'Yes', 'no', 'yes'
# 
# No HR specialists found. 
# 
# It is also not clear how product quality is controlled. Since this is a production company, theoretically there should be positions responsible for product quality (quality engineers etc.)

# In[20]:


#Remove the extra white spaces and bring the Employee_Name and ManagerName columns to the same format.
#transform the values of the TermReason and HispanicLatino columns to lowercase.

df['Employee_Name'] = [" ".join(n.split(',')[::-1]) for n in df['Employee_Name']]


# In[21]:


df['Employee_Name'] = df['Employee_Name'].replace("\s+", " ", regex=True).str.strip()


# In[22]:


df['ManagerName'] = df['ManagerName'].str.replace('.','', regex=True)


# In[23]:


df['Department'] = df['Department'].str.strip()


# In[24]:


df['Position'] = df['Position'].str.strip()


# In[25]:


df['TermReason'] = df['TermReason'].str.lower()


# In[26]:


df['HispanicLatino'] = df['HispanicLatino'].str.lower()


# In[27]:


df


# ## DATA VISUALIZATION

# In[28]:


#For convenience of analysis, we divide the employees into active and terminated. 
#Add employee age and length of service (the number of years an employee has been working for the company).


# In[29]:


df_active = df.query('EmploymentStatus=="Active"')
df_active


# In[30]:


df_terminated = df.query('EmploymentStatus!="Active"')
df_terminated


# In[31]:


df_active['Age'] =df_active['LastPerformanceReview_Date'].dt.year.max() - df_active['DOB'].dt.year


# In[32]:


df_active['LengthOfService'] = df_active['LastPerformanceReview_Date'].dt.year.max() - df_active['DateofHire'].dt.year


# In[33]:


fig = go.Figure(go.Choropleth(locationmode = 'USA-states', 
                              name='State/#Employees/Dept.',  
                              locations = df_active.groupby('State').Employee_Name.count().index,
                              z = df_active.groupby('State').Employee_Name.count().values, 
                              text = df_active.groupby('State').Department.unique(), 
                              colorscale = 'oryel', 
                              colorbar=dict(title='number of<br>employees', 
                                            title_font_size=10, thickness=15, 
                                            tickmode='array',
                                            tickvals=[df_active.groupby('State').Employee_Name.count().min(),
                                                      df_active.groupby('State').Employee_Name.count().max()],  
                                            tickfont_size=8, ticks='outside')))
                
fig.add_scattergeo(
    locationmode='USA-states',
    locations=df_active.groupby('State').Employee_Name.count().index,
    text=df_active.groupby("State").Employee_Name.count(),
    mode='text', hoverinfo='skip')                
                
fig.update_layout(title = 'Company Geography', 
                  title_x=0.5, 
                  margin=dict(t=50, l=0, r=0, b=0), 
                  geo = dict(scope='usa'))

fig


# ### Now, some charts for the most relevant information of the data

# ### About Employee's gender

# In[34]:


df['Sex']


# In[35]:


df['Sex'].unique()


# In[36]:


df['Sex'].value_counts


# In[37]:


df.Sex.unique()


# In[38]:


df.Sex.value_counts().plot(kind="bar", color = 'b')
plt.xlabel('Employees gender')
plt.ylabel('Values')
plt.show()


# ### About worker's marital status

# In[39]:


df.MaritalDesc.value_counts()


# In[40]:


df.MaritalDesc.value_counts().plot(kind="bar")
plt.xlabel('Marital Status')
plt.ylabel('Values')

plt.show()


# ### About the results of the employee satisfaction survey

# In[41]:


df.EmpSatisfaction.value_counts()


# In[42]:


plt.figure(figsize=(12,6))
sns.countplot(x='EmpSatisfaction',data=df,saturation=1)
print(df['EmpSatisfaction'].mean())


# In[43]:


DepartmentvsEmpSatisfaction = pd.crosstab(df.Department, df.EmpSatisfaction)

#Plotting a bar chart.
ax = DepartmentvsEmpSatisfaction.plot(kind='bar',figsize=(15,30),stacked= True, rot =0, label = True)
for c in ax.containers:
    ax.bar_label(c, label_type='center')


# ### Number of employees by departments

# In[44]:


df.Department.value_counts().plot(kind="bar")
plt.xlabel('Deparments')
plt.ylabel('Values')

plt.show()


# In[45]:


df.Department.value_counts()


# In[46]:


df.Department.unique()


# In[47]:


EmploymentStatusvsDepartment = pd.crosstab(df.EmploymentStatus, df.Department)

#Plotting a bar chart.
ax = EmploymentStatusvsDepartment.plot(kind='bar',figsize=(15,30),stacked= True, rot =0, label = True)
for c in ax.containers:
    ax.bar_label(c, label_type='center')


# ### About the recruitment source 

# In[48]:


l=df['RecruitmentSource'].value_counts()

plt.barh(l.index,l,color='y')
plt.title("Sources of Recruitment",fontsize=10)

plt.xlabel('Number of candidates hired')
plt.ylabel('RecruitmentSource')

plt.grid()
plt.legend()
plt.show()


# ### About the Employment Status

# In[49]:


df.EmploymentStatus.value_counts().plot(kind="bar", color = 'green')
plt.xlabel('Employment Status')
plt.ylabel('Values')

plt.show()


# In[50]:


EmploymentStatusvsRecruitmentScore = pd.crosstab(df.EmploymentStatus, df.RecruitmentSource)

#Plotting a bar chart.
ax = EmploymentStatusvsRecruitmentScore.plot(kind='bar',figsize=(15,30),stacked= True, rot =0, label = True)
for c in ax.containers:
    ax.bar_label(c, label_type='center')


# In[51]:


EmploymentStatusvsPerformanceScore = pd.crosstab(df.EmploymentStatus, df.PerformanceScore)

#Plotting a bar chart.
ax = EmploymentStatusvsPerformanceScore.plot(kind='bar',figsize=(15,30),stacked= True, rot =0, label = True)
for c in ax.containers:
    ax.bar_label(c, label_type='center') 


# ### Top 10 Highest VS lowest salaries 

# In[52]:


c=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
x= df['Salary'].sort_values(ascending=False).head(15)
y= df['Salary'].sort_values(ascending=False).tail(15)

plt.bar(c,x,color='b',label="Highest salaries")
plt.bar(c,y,color='r',label="Lowest salaries")

plt.title("Top 10 Highest VS lowest salaries",fontsize=20)

plt.xticks(c)
plt.ylabel('Salaries')
plt.legend()
plt.show()


# ### About the number of absences

# In[53]:


df.Absences.value_counts().plot(kind="bar", color = 'purple')
plt.xlabel('Absences')
plt.ylabel('Values')

plt.show()


# In[54]:


DayAbs = df.groupby(['Department'])['Absences'].sum()
NbEmp = df.groupby(['Department'])['EmpID'].count()

EmpAbs = pd.merge(right=df.groupby(['Department'])[['EmpID']].count().reset_index(), left=df.groupby(['Department'])[['Absences']].sum().reset_index(), on='Department', how="outer")

EmpAbs['AbsRatio'] = round(EmpAbs['Absences']/EmpAbs['EmpID'], 2)



# In[55]:


EmpAbs.sort_values(by=['AbsRatio'], ascending=False, inplace=True)
plt.figure(figsize=(8, 5))
sns.barplot(EmpAbs, x='AbsRatio', y='Department', width=0.7, orient='h')
plt.title("Total of absences by departments")
plt.xlabel('Absences');


# In[56]:


EmpAbs.value_counts()


# In[57]:


AbsPrSex = df.groupby(['Sex'])[['Absences']].sum().reset_index()
plt.figure(figsize=(5, 5))
sns.barplot(AbsPrSex, x='Sex', y='Absences', width=0.5);


# In[58]:


AbsPrSex


# ### Performance Score

# In[59]:


df.PerformanceScore.value_counts()


# In[60]:


df.PerformanceScore.value_counts().plot(kind="bar", color = 'brown')
plt.xlabel('Performance Score')
plt.ylabel('Values')

plt.show()


# In[61]:


df1=df['PerformanceScore'].value_counts()
df1


# In[62]:


plt.plot(df1,'r^-',linewidth=3,ms=9,mfc='w',mec='b')

plt.title('Performance Score of Employees',fontsize=15)
plt.xlabel("Performance Score",fontsize=12)
plt.ylabel("Values",fontsize=12)

plt.grid()
plt.legend()
plt.show()


# In[63]:


df1 = pd.read_csv("hrdata.csv")
df1.Department.unique()


# In[64]:


DepartmentvsEmpSatisfaction = pd.crosstab(df1.Department, df1.EmpSatisfaction)

#Plotting a bar chart.
ax = DepartmentvsEmpSatisfaction.plot(kind='bar',figsize=(15,30),stacked= True, rot =0, label = True)
for c in ax.containers:
    ax.bar_label(c, label_type='center')
    


# In[65]:


df.plot.scatter(y="RecruitmentSource",x="PerformanceScore")


# In[66]:


df.plot.scatter(y="SpecialProjectsCount",x="PerformanceScore")


# In[67]:


df.plot.scatter(y="EmpSatisfaction",x="PerformanceScore")


# In[68]:


df.plot.scatter(y="Department",x="PerformanceScore")


# ### About the employees by age

# In[69]:


fig = px.bar(df_active.groupby('Age').Age.count(),
             color=df_active.groupby('Age').Age.count(), 
             text=df_active.groupby('Age').Age.count(),
             title='Number of Active Employees by Age',
             labels={'color':'number of<br>employees',
                     'value':'number of employees', 
                     'index': 'age'})

fig.update_traces(textfont_size=10, textangle=0, 
                  textposition='outside',
                  hovertemplate = 'age : %{x}<br>number of employees : %{y}') 

fig.update_layout(title_x=0.7, 
                  coloraxis_colorbar_thickness=15,
                  margin=dict(t=50, l=0, r=0, b=0))

fig


# ### About the Salary

# In[70]:


#Plot a histrogram to understand the distribution of the salaries.

plt.figure(figsize = (15,7))
ax = sns.histplot(df.Salary)
for c in ax.containers:
   ax.bar_label(c,label_type = 'edge')
   
#Most of the salaries are centered between $60,000 - $80,000


# In[71]:


bins=[40000,55000,70000,85000,100000,120000]
sns.distplot(df.Salary,bins=bins,color="blue",kde=False)
plt.title("Salaries of workers")
plt.xlabel("Salary")
plt.ylabel("Count")
plt.tight_layout()
plt.grid(True)
plt.show()


# In[72]:


#What is the average of the salaries by departments? The below graph depicts that.
MeanOfSalaries = df.groupby('Department')['Salary'].mean()

ax1 = MeanOfSalaries.plot(kind = "barh")
for c in ax1.containers:
    ax1.bar_label(c,label_type = 'edge')


# In[73]:


#Understanding the differences in pay by gender by plotting a bell curve. 
Male = df[df['Sex'] == "M"]
Female = df[df['Sex'] == "F"]

ggplot(data = df, mapping = aes(x = 'Salary', fill = 'Sex')) +  geom_density()


# From the above it is clear that Males are paid more than Female .

# In[74]:


#Average salaries for Male and Female to find the unadjusted Pay gap for Males and Females. 

AvgSalariesBySex = df.groupby('Sex')['Salary'].mean()
print(AvgSalariesBySex)

unadjusted_pay_gap = 70629.4 - 67786.7
print("Unadjusted pay gap for Females is",unadjusted_pay_gap)


# This means the organization pays females $2843 less than men. However, this metric is unadjusted for various factors that are known to affect salary, including job level, tenure, previous work experience, and more. 

# In[75]:


df.plot.scatter(x="Salary",y="PerformanceScore")


# In[76]:


df.plot.scatter(x="Salary",y="EmpSatisfaction")


# In[77]:


df.plot.scatter(x="Salary",y="RecruitmentSource")


# In[78]:


df.plot.scatter(x="Salary",y="EngagementSurvey")


# In[79]:


df.plot.scatter(x="Salary",y="Department")


# ### About the 'Days late in the last 30 days'

# In[80]:


sns.barplot(round(df.groupby(['Department'])[['DaysLateLast30']].mean().reset_index(), 2).sort_values(by=['DaysLateLast30'], ascending=False), x='DaysLateLast30', y='Department', width=0.7, orient='h')
plt.xlabel('Days late in the last 30 days');


# In[81]:


#The below code assigns department names to the department ID's.
# Dept ID 1= HR, 2 = IT, 3= Operations, 4= Marketing, 5= Accounting, 6=Sales. 

df1 = pd.read_csv("hrdata.csv")
 
def AssignDeptNames(row):
    if row == 1:
        return 'Human Resources'
    elif row == 2:
        return 'Information Technology'
    elif row == 3:
        return 'Operations'
    elif row ==4:
        return 'Marketing'
    elif row == 5:
        return 'Accounting'
    else:
        return 'Sales'
    
df1['DeptName'] = df1['DeptID'].map(AssignDeptNames)


# In[82]:


#Find the unique values in the Performance Scores column. 
df1.PerformanceScore.unique()


# In[83]:


# Assign a numeric value to the Performance Scores column.
# Exceed = 5, Fully Meets = 3, Needs Improvement = 1, PIP = 0

def PerformanceNumericLabels(columnname):
    if columnname =='Exceeds':
        return 5
    elif columnname == 'Fully Meets':
        return 3
    elif columnname== 'Needs Improvement':
        return 1
    else:
        return 0

df1['PerformanceNumericLabels'] = df1['PerformanceScore'].map(PerformanceNumericLabels)
#display(df1.head())


# In[84]:


# From which Job source are we getting high performers(Exceeds) and Low Performers(Needs Improvements, PIP)
JobSourceVPerfScore = pd.crosstab(df1.RecruitmentSource, df1.PerformanceScore)

#Plotting a bar chart.
ax = JobSourceVPerfScore.plot(kind='bar',figsize=(15,10),stacked= True, rot =0, label = True)
for c in ax.containers:
    ax.bar_label(c, label_type='center')
    
#The above code plots a stacked bar graph for RecruitmentSource and PerformanceScore.Something very evident is employees with Performance score 
#of "Fully Meets" are coming from Indeed and LinkedIn.
#Look at the Performance score Legend on the right. 


# ### What is the Staff Turnover in the Company?

# In[85]:


total=0
total_list=[]
hired_list=[]
left_list=[]
mean_list=[]
turnover_list=[]
df['YearofHire'] = df['DateofHire'].dt.year
df['YearofTermination'] = df['DateofTermination'].dt.year

for year in df['YearofHire'].sort_values().unique():
    
    hired = df.query('YearofHire == @year').Employee_Name.count()
    left = df.query('YearofTermination == @year').Employee_Name.count()
    total = total + hired - left
    total_list.append(total)
    hired_list.append(hired)
    left_list.append(left)
    
for i in range(len(total_list)):
    if i==0:
        mean = (0 + total_list[i]) / 2
        mean_list.append(mean)
    else:
        mean = (total_list[i-1] + total_list[i]) / 2
        mean_list.append(mean)
        
    turnover = ((left_list[i] / mean_list[i])*100).round(2) 
    turnover_list.append(turnover)
    

print('Total at the end of each year : \n', total_list)
print('Terminated each year : \n', left_list)
print('Hired each year : \n', hired_list)
print('Average number of staff per year : \n', mean_list)  
print('Staff turnover per year : \n', turnover_list)


# In[ ]:


x = df['YearofHire'].sort_values().unique()

fig = make_subplots(rows=2, cols=1, row_heights=[0.2, 0.8], 
                    vertical_spacing=0.01, shared_xaxes=True)

fig.add_trace(go.Scatter(name="Staff Turnover", 
                         x=x, y=turnover_list, mode="lines+markers+text",
                         text=turnover_list, textposition="middle left", 
                         marker_color='magenta'), 
             1, 1)

fig.add_trace(go.Bar(name="Staff Hired", 
                     x=x, 
                     y=hired_list,
                     text=hired_list,
                     textposition='outside',
                     marker_color='green'),  
             2, 1)

fig.add_trace(go.Bar(name="Staff Total", 
                     x=x, 
                     y=total_list,
                     text=total_list,
                     textposition='outside',
                     marker_color='cadetblue'),  
             2, 1)

fig.add_trace(go.Bar(name="Staff Terminated", 
                     x=x, 
                     y=left_list,
                     text=left_list,
                     textposition='outside',
                     marker_color='red'),  
             2, 1)

fig.update_traces(textfont_size=10)

fig.update_layout(title='Annual Staff Turnover from 2006 to 2018',
                  title_x=0.5, margin=dict(t=50, l=0, r=0, b=0))

fig


# ## Build & Training a Machine Learning Algorithm:

# In[86]:


df.isnull().sum()


# In[87]:


#Getting column names for individual exploration
df.columns


# In[88]:


df['Sex'].unique()


# In[89]:


df['Department'].unique()


# In[90]:


df['Position'].unique()


# In[91]:


df['RecruitmentSource'].unique()


# In[92]:


df.dropna(how='all', inplace=True)


# In[93]:


#Exploring performance score and it's ID
df[['PerformanceScore','PerfScoreID']]


# In[94]:


print(df['PerformanceScore'].unique())
print(df['PerfScoreID'].unique())


# In[95]:


#Let's now drop some columns that are irrelevant
df.drop(['DaysLateLast30','LastPerformanceReview_Date',
         'DateofTermination','TermReason','DaysLateLast30','Zip'],axis=1,inplace=True)


# ### Creating the machine learning model

# In[96]:


#select only necessary columns
df_select = df[['MaritalDesc','Sex', 'EmploymentStatus', 'Department', 'PerformanceScore',  'Position', 'CitizenDesc', 'HispanicLatino',
          'RaceDesc', 'ManagerName', 'RecruitmentSource', 'EmpSatisfaction', 'SpecialProjectsCount','Salary', 'Absences', 'Termd']]


# In[97]:


df_select


# In[98]:


df_select.head()


# In[99]:


df_select.info()


# In[100]:


df_select.isnull().sum()


# In[101]:


#Select only Categorical features
df_select.columns[:-4]


# In[102]:


#Apply get dummies
df_dummies = pd.get_dummies(df_select, columns=df_select.columns[:-4], drop_first=True, dtype=float)
df_dummies


# In[103]:


df_dummies.info()


# ### Prediction

# In[104]:


##Spliting Data


# In[105]:


# drop the 'Termd' column from the DataFrame to create the feature matrix
x = df_dummies.drop('Termd', axis=1)
# create the target vector
y = df_dummies['Termd']


# In[106]:


x


# In[107]:


y


# In[108]:


# split the data into training and testing subsets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=22)


# In[109]:


# check shape
print('x_train: ', x_train.shape)
print('x_test: ', x_test.shape)
print('y_train: ', y_train.shape)
print('y_test: ', y_test.shape)


# In[110]:


# make a balanced dataset by using SMOTE technique.
sm = SMOTE(random_state=22)
# Create a SMOTE object
oversample = SMOTE()
# Fit the SMOTE object to the training data and oversample the minority class
x_train, y_train = oversample.fit_resample(x_train, y_train)
x_train


# In[111]:


# create an instance of the StandardScaler
scaler = StandardScaler()
scaler.fit(x_train)


# In[112]:


# fit the scaler to the training data X_train
scaler = StandardScaler()
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_test


# ### Build prediction Models

# In[113]:


# create a dictionary of models
models = {
    "   K-NearestNeighbors": KNeighborsClassifier(),
    "   LogisticRegression": LogisticRegression(),
    "SupportVectorMachine": SVC(),
    "         DecisionTree": DecisionTreeClassifier(),
    "        NeuralNetwork": MLPClassifier(),
    "         RandomForest": RandomForestClassifier(n_estimators=500),
    "         XGBClassifier": XGBClassifier(n_estimators=700)
}

# loop through the models and fit each one to the training data
for name, model in models.items():
    model.fit(x_train, y_train)
    print(name + " trained.")


# In[114]:


# loop through the models and make predictions on the test data
for name, model in models.items():
    print(name + " Accuracy: {:.2f}%".format(model.score(x_test, y_test) * 100))
    y_pred = model.predict(x_test)
    
  # plot the confusion matrix as a heatmap
    cm = confusion_matrix(y_test, y_pred)
    ax = sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
    ax.set_title(name + " Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    plt.show()
    
    # print the classification report for each model
    report = classification_report(y_test, y_pred)
    print(name + " Classification Report:")
    print(report)   
    
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))


# In[115]:


from sklearn.feature_selection import f_classif, chi2, mutual_info_classif
from sklearn.feature_selection import SelectKBest


# In[116]:


selector = SelectKBest(k=10, score_func=f_classif)


# In[117]:


selector.fit(x_train, y_train)


# In[118]:


selector.get_support(indices=True)


# In[119]:


x.columns[selector.get_support(indices=True)]


# In[120]:


Select_columns = pd.DataFrame({'Important_Feature':x.columns[selector.get_support(indices=True)],
                  'Score':selector.get_support(indices=True)} )


# In[121]:


plt.figure(figsize=(10,5))
sns.barplot(x='Score', y='Important_Feature', data=Select_columns)


# In[122]:


rf_model = MLPClassifier()
rf_model.fit(x_train, y_train)


# In[123]:


#export and save my prediction model
import pickle
from array import array

pickle.dump(rf_model, open('performanceScoremodel3.pkl', 'wb'))

