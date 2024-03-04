#!/usr/bin/env python
# coding: utf-8

# In[98]:


import pandas as pd


# In[99]:


#Visualize the dataset
df_vis = pd.read_csv(r'F:\Project_files\output.csv')




# In[100]:


df_vis.hist(figsize=[15,15])


# In[1]:


#Step-1: Data Preprocessing Steps are shown here


# In[5]:


#we have merged the contents of  our whole dataset into file whose name is file-2 which would be our new dataset file
import pandas as pd

# Load your data into a pandas DataFrame
df = pd.read_csv(r'F:\Project_files\file-2.csv')

#This will display the contents of the file 
df




# In[4]:


#This code is used to know how many distinct values are there in the file-2 which our dataset file
import pandas as pd

# Load your data into a pandas DataFrame
df = pd.read_csv(r'F:\Project_files\file-2.csv')

# Use the `nunique()` function to find the distinct values for all columns in the dataset
distinct_values = df.nunique()

# Print the result
print(distinct_values)


# In[6]:


#Now we need to convert our activity column to categorical value 
#Experimenting with Attributes Combinations values
import pandas as pd

columns=['activity']

for col in columns:
    df[col]=df[col].astype('category')

    


# In[6]:


#Through this you can see that datatype of activity column has been changed to categorical value
df.dtypes 


# In[7]:


#This will assign unique code values to the categorical data
#Handling Text and Categorical attributes are performed here
for col in columns:
    df[col] = df[col].astype('category').cat.codes
#It will display Unique code values in activity column
print(df.head(3)) 



# In[8]:


#To check if there are any missing or NAN values
print(df.isna().sum())


# In[9]:


#Performed Correlation as well as handled missing values in the column 
#As well performed Data-Cleaning by replacing the NAN values with value 0 
df.corr(method="pearson").fillna(0)
print(df.head(3))


# In[10]:


#It will remove the outliers and it will only display the data which falls under the category of 1QR Quantile range
import pandas as pd


# Calculate the IQR (interquartile range) of the data
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1

# Remove the outliers
df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]

# Display the cleaned data
df.head(3)


# In[11]:


#Here, Feature Scaling is done using StandardScaler class
#It will display scaling features
#It first calculates the mean and standard deviation of each feature in the DataFrame and then scales the values accordingly.

import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the dataset
#df = pd.read_csv('data.csv')

# Instantiate the StandardScaler object
scaler = StandardScaler()

# Fit and transform the data
df_scaled = scaler.fit_transform(df)

# Convert the result to a pandas DataFrame
df_scaled = pd.DataFrame(df_scaled, columns=df.columns)

# Print the scaled data
#It will display first few rows of the scaled Dataframe
print(df_scaled.head(3))



# In[12]:


#Here, we will be using LabelEncoder to perform Transform feature encoding
#It will assign unique codes to the columns listed in the cat_cols
#Here, we would loop through categorical column and use the fit_transform method of the encoder object to encode the values in the column. 
#The resulting encoded values replace the original values in the dataframe.

import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the dataset
#df = pd.read_csv('data.csv')

# Select the categorical columns
cat_cols = ['activity']

# Instantiate the LabelEncoder object
encoder = LabelEncoder()

# Encode the categorical columns
for col in cat_cols:
    df[col] = encoder.fit_transform(df[col])

# Print the encoded data
print(df.head(3))


# In[12]:


#Here, you can see that there the values have been updated
#Now you can use this data for ML
df


# In[13]:


#This will store the result of feature encoding into a new csv file and it can be applied for ML algorithms 
import pandas as pd


# Store the dataframe into a csv file
df.to_csv('F:\Project_files\output-1.csv', index=False)


# In[14]:


df



# #Step-2: ML Classification is performed over here where it does Classification using 5 types of models
# 

# In[15]:


#Here we are importing the necessary packages which are required for modeling are ML model
#1. We are using KNN Model for Modeling
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score




# In[18]:


# Load the dataset and reduced the number of rows by using this formula i.e it reduced to 419887 out of 1048575
df = pd.read_csv('F:\Project_files\output.csv', nrows=int(0.1*sum(1 for line in open('F:\Project_files\output.csv'))))
#df=df.head()

# Split the dataset into features and target labels
X = df.drop('activity', axis=1)
y = df['activity']

# Preprocess the data
X.fillna(X.mean(), inplace=True)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


df


# In[19]:


#Here we have taken the value of k from 1to5
# Checking the accuracy for Full-Feature Selection
k = [1,2,3,4,5] #random values 
for i in k:
   knn = KNeighborsClassifier(n_neighbors=i, metric='euclidean')
   knn.fit(X_train, y_train)
   y_pred = knn.predict(X_test)
   acc = accuracy_score(y_test, y_pred)
   print(f"Accuracy: {acc}","", i)


# In[20]:


#We are doing feature Selection so we have to find the correlationship between each column with target attribute
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read in the dataset
df = pd.read_csv('F:\Project_files\output.csv')

# Calculate the correlation matrix
corr_matrix = df.corr()
fig, ax = plt.subplots(figsize=(30, 20))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)



# Show the plot
plt.show()


# In[21]:


#We have selected the features which are very much common to the target feature
df_unique = pd.DataFrame(df, columns=['activityChange','lastSensorLocation','lastSensorEventSeconds'])
df_unique


# In[22]:


#We would be training data for the feature selection based on common features and target features
X = df_unique
y = df['activity']

# Preprocess the data
X.fillna(X.mean(), inplace=True)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[25]:


#We would be finding the accuracy of the model for feature selection
#Accuracy of feature selection and full-feature may vary because we have not included all the columns 
#We have only taken those columns which are very much common based on target field
k = [1,2,3,4,5] #random values 
for i in k:
   knn = KNeighborsClassifier(n_neighbors=i, metric='euclidean')
   knn.fit(X_train, y_train)
   y_pred = knn.predict(X_test)
   acc = accuracy_score(y_test, y_pred)
   print(f"KNN Accuracy Score: {acc}","", i)


# In[40]:


#2. SVM Implementation
from sklearn.svm import SVC


# In[ ]:


# Load the dataset and reduced the number of rows by using this formula i.e it reduced to 419887 out of 1048575
df = pd.read_csv('F:\Project_files\output.csv', nrows=int(0.1*sum(1 for line in open('F:\Project_files\output.csv'))))
#df=df.head()

# Split the dataset into features and target labels
X = df.drop('activity', axis=1)
y = df['activity']


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[ ]:


# Train the SVM model
svm = SVC(kernel='rbf', C=1, gamma='scale')
svm.fit(X_train, y_train)

# Test the SVM model
y_pred = svm.predict(X_test)

# Evaluate the performance of the model
accuracy = accuracy_score(y_test, y_pred)
print("SVM Accuracy Score:", accuracy)


# In[43]:


#3. Random Forest Implementation
from sklearn.ensemble import RandomForestClassifier


# In[44]:


# Load the dataset and reduced the number of rows by using this formula i.e it reduced to 419887 out of 1048575
df = pd.read_csv('F:\Project_files\output.csv', nrows=int(0.1*sum(1 for line in open('F:\Project_files\output.csv'))))
#df=df.head()

# Split the dataset into features and target labels
X = df.drop('activity', axis=1)
y = df['activity']


# In[45]:


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the Random Forest classifier
rf.fit(X_train, y_train)

# Predict the labels of the test data
y_pred = rf.predict(X_test)


# In[46]:


# Evaluate the performance of the classifier
accuracy = accuracy_score(y_test, y_pred)
print("Random forest Accuracy Score:", accuracy)


# In[35]:


#4. Naive Bayes Implementation


# In[47]:


# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
knn = KNeighborsClassifier(n_neighbors=300)
knn.fit(X_train, y_train)

pred = knn.predict(X_test)


# In[48]:


from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)
from sklearn import metrics
print("Gaussian Naive Bayes model accuracy(in %):", metrics.accuracy_score(y_test, y_pred)*100)
print('Training set score: {:.4f}'.format(gnb.score(X_train, y_train)))

print('Test set score: {:.4f}'.format(gnb.score(X_test, y_test)))


# In[38]:


#5. Decision Tree Implementation



# In[49]:


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support

# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Decision Tree Accuracy:", metrics.accuracy_score(y_test, y_pred))

# Calculate precision and recall for each label in your dataset
precision, recall, _, _ = precision_recall_fscore_support(y_test, y_pred)

# Print precision and recall for each label
for i in range(len(precision)):
    print("Label {}: Precision: {:.3f}, Recall: {:.3f}".format(i, precision[i], recall[i]))

# Calculate the average precision and recall across all labels
avg_precision = np.mean(precision)
avg_recall = np.mean(recall)

print("Average Precision: {:.3f}, Average Recall: {:.3f}".format(avg_precision, avg_recall))


# #Step-3: Evaluation and Analysis of each model

# In[50]:


#1.Evaluation Analysis for Decision Tree
from sklearn.tree import DecisionTreeClassifier
# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)


# In[51]:


# Evaluation metrics 1
from sklearn import metrics
print("Decision-tree Accuracy Score:",metrics.accuracy_score(y_test, y_pred))


# In[52]:


#Evaluation metrics 2
from sklearn.metrics import f1_score
f = f1_score(y_true = y_test , y_pred = y_pred,average = 'weighted')   
print(f)


# In[53]:


#Evaluation metrics 3
from sklearn.metrics import precision_score
print('Precision: %.3f' % precision_score(y_test, y_pred, average = 'weighted'))


# In[54]:


#Evaluation metrics 4
from sklearn.metrics import recall_score	
print('Recall: %.3f' % recall_score(y_test, y_pred, average = 'weighted'))


# In[55]:


#Visualization Technique-1
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


# In[56]:


cm_df = pd.DataFrame(cm)
cm_df


# In[57]:


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# In[58]:


pip install scikit-plot


# In[59]:


#Visualization technique-2
#define metrics
import matplotlib.pyplot as plt
import scikitplot as skplt
from sklearn.metrics import roc_curve
Y_test_probs = clf.predict_proba(X_test)

skplt.metrics.plot_roc_curve(y_test, Y_test_probs,
                       title="Digits ROC Curve", figsize=(12,6))


# In[60]:


#Vistualization technique-3
skplt.metrics.plot_precision_recall_curve(y_test, Y_test_probs,
                       title="Digits Precision-Recall Curve", figsize=(12,6));


# In[61]:


#2.Evaluation Analysis for KNN
from sklearn.neighbors import KNeighborsClassifier


# In[62]:


#Evaluation metrics-1
k = [1,2,3,4,5] #random values 
for i in k:
   knn = KNeighborsClassifier(n_neighbors=i, metric='euclidean')
   knn.fit(X_train, y_train)
   y_pred = knn.predict(X_test)
   acc = accuracy_score(y_test, y_pred)
   print(f"KNN Accuracy Score: {acc}","", i)


# In[63]:


#Evaluation metrics 2
from sklearn.metrics import f1_score
f = f1_score(y_true = y_test , y_pred = y_pred,average = 'weighted')   
print(f)


# In[64]:


#Evaluation metrics 3
from sklearn.metrics import precision_score
print('Precision: %.3f' % precision_score(y_test, y_pred, average = 'weighted'))


# In[65]:


#Evaluation metrics 4
from sklearn.metrics import recall_score	
print('Recall: %.3f' % recall_score(y_test, y_pred, average = 'weighted'))


# In[66]:


##Visualization Technique-1
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


# In[67]:


cm_df = pd.DataFrame(cm)
cm_df


# In[68]:


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# In[69]:


##Visualization Technique-2
import matplotlib.pyplot as plt
import scikitplot as skplt
from sklearn.metrics import roc_curve
Y_test_probs = clf.predict_proba(X_test)

skplt.metrics.plot_roc_curve(y_test, Y_test_probs,
                       title="Digits ROC Curve", figsize=(12,6))


# In[60]:


#Visualization Technique-3
skplt.metrics.plot_precision_recall_curve(y_test, Y_test_probs,
                       title="Digits Precision-Recall Curve", figsize=(12,6));


# In[70]:


#3.Evaluation Analysis for Random-Forest
from sklearn.ensemble import RandomForestClassifier


# In[71]:


# Initialize the Random Forest classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the Random Forest classifier
rf.fit(X_train, y_train)

# Predict the labels of the test data
y_pred = rf.predict(X_test)


# In[72]:


# Evaluate metrics 1
accuracy = accuracy_score(y_test, y_pred)
print("Random forest Accuracy Score:", accuracy)


# In[73]:


#Evaluation metrics 2
from sklearn.metrics import f1_score
f = f1_score(y_true = y_test , y_pred = y_pred,average = 'weighted')   
print(f)


# In[74]:


#Evaluation metrics 3
from sklearn.metrics import precision_score
print('Precision: %.3f' % precision_score(y_test, y_pred, average = 'weighted'))


# In[76]:


#Evaluation metrics 4
from sklearn.metrics import recall_score	
print('Recall: %.3f' % recall_score(y_test, y_pred, average = 'weighted'))


# In[77]:


##Visualization Technique-1
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


# In[78]:


cm_df = pd.DataFrame(cm)
cm_df


# In[79]:


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# In[80]:


##Visualization Technique-2
import matplotlib.pyplot as plt
import scikitplot as skplt
from sklearn.metrics import roc_curve
Y_test_probs = clf.predict_proba(X_test)

skplt.metrics.plot_roc_curve(y_test, Y_test_probs,
                       title="Digits ROC Curve", figsize=(12,6))


# In[81]:


#Visualization Technique-3
skplt.metrics.plot_precision_recall_curve(y_test, Y_test_probs,
                       title="Digits Precision-Recall Curve", figsize=(12,6));


# In[82]:


# 4.Naive Bayes
knn = KNeighborsClassifier(n_neighbors=300)
knn.fit(X_train, y_train)

pred = knn.predict(X_test)


# In[83]:


from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)
from sklearn import metrics
print("Gaussian Naive Bayes model accuracy(in %):", metrics.accuracy_score(y_test, y_pred)*100)
print('Training set score: {:.4f}'.format(gnb.score(X_train, y_train)))

print('Test set score: {:.4f}'.format(gnb.score(X_test, y_test)))


# In[84]:


# Evaluate metrics 1
accuracy = accuracy_score(y_test, y_pred)
print(" Naive Bayes Accuracy Score:", accuracy)


# In[85]:


#Evaluation metrics 2
from sklearn.metrics import f1_score
f = f1_score(y_true = y_test , y_pred = y_pred,average = 'weighted')   
print(f)


# In[86]:


#Evaluation metrics 3
from sklearn.metrics import precision_score
print('Precision: %.3f' % precision_score(y_test, y_pred, average = 'weighted'))


# In[87]:


#Evaluation metrics 4
from sklearn.metrics import recall_score	
print('Recall: %.3f' % recall_score(y_test, y_pred, average = 'weighted'))


# In[88]:


##Visualization Technique-1
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


# In[89]:


cm_df = pd.DataFrame(cm)
cm_df


# In[90]:


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# In[81]:


##Visualization Technique-2
import matplotlib.pyplot as plt
import scikitplot as skplt
from sklearn.metrics import roc_curve
Y_test_probs = clf.predict_proba(X_test)

skplt.metrics.plot_roc_curve(y_test, Y_test_probs,
                       title="Digits ROC Curve", figsize=(12,6))


# In[82]:


#Visualization Technique-3
skplt.metrics.plot_precision_recall_curve(y_test, Y_test_probs,
                       title="Digits Precision-Recall Curve", figsize=(12,6));


# In[91]:


df_eval = pd.read_csv(r'C:\Users\kunal\OneDrive\Desktop\csh101\EvalRes.csv')


# In[92]:


import matplotlib.pyplot as plt


# In[93]:


df_eval.plot.line(x='Algorithm',y='Accuracy')


# In[94]:


df_eval.plot.line(x='Algorithm',y='Precision')


# In[95]:


df_eval.plot.line(x='Algorithm',y='Recall')


# In[96]:


df_eval.plot.line(x='Algorithm',y='F1-score')


# In[97]:


df_eval.plot.line(x='Algorithm',y='AUC-ROC')


# In[1]:


import matplotlib.pyplot as plt
import math


# In[2]:


x = ['Full dataset', 'Reduced featureset']
y = [94, 97]
low = min(y)
high = max(y)
plt.ylim([math.ceil(low-0.5*(high-low)), math.ceil(high+0.5*(high-low))])
plt.bar(x,y) 

plt.xlabel('KNN')
plt.ylabel('Accuracy (%)')
plt.show()


# In[3]:


import matplotlib.pyplot as plt
import math
x = ['Full dataset', 'Reduced featureset']
y = [95, 97]
low = min(y)
high = max(y)
plt.ylim([math.ceil(low-0.5*(high-low)), math.ceil(high+0.5*(high-low))])
plt.bar(x,y) 

plt.xlabel('Random Forest')
plt.ylabel('Accuracy (%)')
plt.show()


# In[ ]:




