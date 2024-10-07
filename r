*****************************                               Gaussian Mixture Model Clustering                               *******************************

!pip install matplotlib

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

dataset = pd.read_csv("Clustering_gmm.csv")

dataset.head(10)

plt.figure()
plt.scatter(dataset["Weight"],dataset["Height"])
plt.xlabel("Weight")
plt.ylabel("Height")

kmeans = KMeans(n_clusters = 4)
kmeans.fit(dataset)

predictions = kmeans.predict(dataset)

predictions

dataframe1 = pd.DataFrame(dataset)
dataframe1

dataframe1["predictions"] = predictions
dataframe1

color = ["red", "yellow", "green", "blue"]

for i in range(0,4):
    data = dataframe1[dataframe1["predictions"] == i]
    plt.scatter(data["Weight"], data["Height"], c = color[i])
plt.show()

#GMM

dataset_gmm = pd.read_csv("Clustering_gmm.csv")

dataset_gmm.head(10)

gmm = GaussianMixture(n_components = 4)
gmm.fit(dataset_gmm)

predictions1 = gmm.predict(dataset_gmm)
predictions1

dataframe2 = pd.DataFrame(dataset_gmm)
dataframe2

dataframe2["predictions1"] = predictions1
dataframe2

color = ["red", "yellow", "green", "blue"]

for i in range(0,4):
    data = dataframe2[dataframe2["predictions1"] == i]
    plt.scatter(data["Weight"], data["Height"], c = color[i])
plt.show()



*******************************                                Decision Tree with EDA                            *************************************

import pandas as pd

df = pd.read_csv("titanic.csv")

df.head(4)

df.info()

df.describe()

df.isnull().sum()

df.hist(bins = 20, figsize = (14,10))
import matplotlib.pyplot as plt
plt.show()

!pip install seaborn

df.drop(["Name"], axis = 1, inplace = True)

from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
df["Sex"] = lb.fit_transform(df["Sex"])

df["Cabin"] = lb.fit_transform(df["Cabin"])
df["Embarked"] = lb.fit_transform(df["Embarked"])
df["Ticket"] = lb.fit_transform(df["Ticket"])

corr_matrix = df.corr()
plt.figure(figsize = (12,8))
import seaborn as sns
sns.heatmap(corr_matrix, annot = True, cmap = 'coolwarm')
plt.show()

sns.pairplot(df, hue = "Survived" , diag_kind = "hist")

features_e = ["Pclass", "Sex" , "Embarked"]

for feature in features_e:
    sns.countplot(x = feature, hue = "Survived", data = df)
    plt.show()

df = df.drop(["PassengerId", "Ticket", "Cabin"], axis = 1)

df.isnull().sum()

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy = "median")

df["Age"] = imputer.fit_transform(df[["Age"]])

df.isnull().sum()

df.columns

X = df.iloc[:,1:]

X.columns

y = df.iloc[:,0]

from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()

from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(dt,X,y, cv = 5)
cv_scores

import numpy as np

np.mean(cv_scores)

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

cv_scores2 = cross_val_score(dt,X,y, cv = 5)

np.mean(cv_scores2)

from sklearn.model_selection import train_test_split

xtrain,xtest,ytrain,ytest = train_test_split(X, y, test_size = 0.25, random_state = 42)

dt_final = DecisionTreeClassifier()

dt_final.fit(xtrain,ytrain)

preds = dt_final.predict(xtest)

lr_final = LogisticRegression()

lr_final.fit(xtrain,ytrain)
preds_lr = lr_final.predict(xtest)

df["Survived"].value_counts()

from sklearn.metrics import accuracy_score, confusion_matrix

accuracy_score(ytest,preds), accuracy_score(ytest,preds_lr)

confusion_matrix(ytest,preds_lr)

from sklearn.metrics import classification_report

print(classification_report(ytest,preds))

print(classification_report(ytest,preds_lr))



*******************************************                             Gaussian Naive Bayes for discrete data                         *******************************************
from sklearn.datasets import load_breast_cancer

ds = load_breast_cancer()

X = ds.data

y = ds.target

from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import train_test_split

xtrain,xtest,ytrain,ytest = train_test_split(X,y,test_size=0.25,random_state=1)

model1 = GaussianNB()

model1.fit(xtrain,ytrain)

predictions = model1.predict(xtest)

from sklearn.metrics import accuracy_score,confusion_matrix

accuracy_score(ytest,predictions)

confusion_matrix(ytest,predictions)



*******************************************                Gaussian Naive Bayes for contionuous data                    ********************************************

from sklearn.datasets import load_iris

ds = load_iris()

y = ds.target

from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import train_test_split

xtrain,xtest,ytrain,ytest = train_test_split(X,y,test_size=0.25,random_state=1)

model1 = GaussianNB()

model1.fit(xtrain,ytrain)

predictions = model1.predict(xtest)

from sklearn.metrics import accuracy_score,confusion_matrix

accuracy_score(ytest,predictions)

confusion_matrix(ytest,predictions)



*********************************************            Bernoulli/Multinomial Naive Bayes             ************************************************

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.naive_bayes import BernoulliNB

newsgroups = fetch_20newsgroups(subset = 'all')

vectorizer_BNB = CountVectorizer(binary = True)

X1 = vectorizer_BNB.fit_transform(newsgroups.data)

y = newsgroups.target

from sklearn.model_selection import train_test_split

xtrain1,xtest1,ytrain,ytest = train_test_split(X1,y,test_size=0.25,random_state=42)

BNB = BernoulliNB()

BNB.fit(xtrain1,ytrain)

y_pred1 = BNB.predict(xtest1)

from sklearn.metrics import accuracy_score

accuracy_score(ytest,y_pred1)

from sklearn.naive_bayes import MultinomialNB

newsgroups = fetch_20newsgroups(subset = 'all')

vectorizer_MNB = CountVectorizer(binary = False)

X2 = vectorizer_MNB.fit_transform(newsgroups.data)

y = newsgroups.target

from sklearn.model_selection import train_test_split

xtrain2,xtest2,ytrain,ytest = train_test_split(X2,y,test_size=0.25,random_state=42)

MNB = MultinomialNB()

MNB.fit(xtrain2,ytrain)

y_pred2 = MNB.predict(xtest2)

from sklearn.metrics import accuracy_score

accuracy_score(ytest,y_pred2)

#Conclusion: multinomial is better compared to Bernoulli because frequency based vectors gave better accuracy

#Naive Bayes using Multinomial and Tfid Vectorizer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.pipeline import make_pipeline

model = make_pipeline(TfidfVectorizer(),MultinomialNB())

train_data = fetch_20newsgroups(subset = 'train')
test_data = fetch_20newsgroups(subset = 'test')                                

model.fit(train_data.data,train_data.target)

predictions_tf = model.predict(test_data.data)

accuracy_score(test_data.target,predictions_tf)




**************************************************         Perform Binary Classification on Cancer Dataset using Feed Forward Neural Network          **********************************************

!pip install keras
!pip install tensorflow

import keras
import tensorflow

!pip install pandas

!pip install scikit.learn

from sklearn.datasets import load_breast_cancer

dataset = load_breast_cancer()

x = dataset.data

x.shape

y = dataset.target

y.shape

from sklearn.model_selection import train_test_split

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.25,random_state=1)

xtrain.shape

xtest.shape

from keras.models import Sequential

from keras.layers import Dense

nnmodel = Sequential()

nnmodel.add(Dense(18,activation='relu',input_dim=30))
nnmodel.add(Dense(12,activation='relu'))
nnmodel.add(Dense(1,activation='sigmoid'))

nnmodel.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

nnmodel.fit(xtrain,ytrain,epochs=500,batch_size=40)

predictions = nnmodel.predict(xtest)

predictions

ytest

class_labels=[]

predictions.size

for i in range(143):
    if (predictions[i]>=0.5):
        class_labels.append(1)
    else:
        class_labels.append(0)


class_labels

from sklearn.metrics import accuracy_score,confusion_matrix

accuracy_score(ytest,class_labels)

confusion_matrix(ytest,class_labels)


*******************             Performing Regression using Feed Forward Neural Network(csv data)                 ************************

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense

df = pd.read_csv("flower.xls",header=None)
df

x = df.iloc[:,:-1].astype(float)
x

y = df.iloc[:,-1]
y

x.shape

y.shape

from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
y = lb.fit_transform(y)
y

import keras
import tensorflow as tf

encoded_y = tf.keras.utils.to_categorical(y)
encoded_y

from sklearn.model_selection import train_test_split

xtrain,xtest,ytrain,ytest=train_test_split(x,encoded_y,test_size=0.25,random_state=1)

xtrain.shape

xtest.shape

from keras.models import Sequential

from keras.layers import Dense

model = Sequential()

model.add(Dense(8,activation='relu',input_dim=4))
model.add(Dense(6,activation='relu'))
model.add(Dense(3,activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['mae'])

model.fit(xtrain,ytrain,epochs=100,batch_size=10)

predictions = model.predict(xtest)

predictions

for i in range(1,len(predictions),3):
    print(predictions[i],ytest[i])

y_pred = []
for i in range(0,len(predictions)):
    y_pred.append(np.argmax(predictions[i]))

y_test = []
for i in range(0,len(ytest)):
    y_test.append(np.argmax(ytest[i]))


from sklearn.metrics import r2_score

r2_score(y_test,y_pred)



*******************        Performing Regression using Feed Forward Neural Network(random data)           ************************

import numpy as np

np.random.seed(42)

X = np.random.rand(1000,10)

y = np.random.rand(1000)

X.shape

y.shape

from sklearn.model_selection import train_test_split

xtrain,xtest,ytrain,ytest = train_test_split(X,y,test_size=0.25,random_state=1)

xtrain.shape

xtest.shape

from keras.models import Sequential

from keras.layers import Dense

model = Sequential()

model.add(Dense(12,activation='relu',input_dim=10))
model.add(Dense(8,activation='relu'))
model.add(Dense(1,activation='linear'))

model.compile(loss='mse',optimizer='adam',metrics=['mae'])

model.fit(xtrain,ytrain,epochs=400,batch_size=50)

predictions = model.predict(xtest)

from sklearn.metrics import r2_score,mean_absolute_error

r2_score(ytest,predictions)

mean_absolute_error(ytest,predictions)





****************************         Perform Binary Classification on Cancer Dataset using Feed Forward Neural Network (entire data)          ****************************


from sklearn.datasets import load_breast_cancer
df=load_breast_cancer()

x=df.data
y=df.target
x.shape

from keras.models import Sequential
from keras.layers import Dense
model=Sequential()

model.add(Dense(20,activation="relu",input_dim=30))
model.add(Dense(18,activation="relu"))
model.add(Dense(1,activation="sigmoid"))

model.compile(loss="binary_crossentropy",optimizer="adam",metrics=['accuracy'])
model.fit(x,y,batch_size=10,epochs=450,)

prediction=model.predict(x)
prediction

pred=[]
for i in range(prediction.size):
    if(prediction[i]>=0.5):
        pred.append(1)
    else:
        pred.append(0)
pred

from sklearn.metrics import accuracy_score,confusion_matrix
accuracy_score(y,pred)


confusion_matrix(y,pred)


************************************************          Movie Recommendation System (Content Based Filtering)*             ***********************************************

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv('movies.csv')
df

df.columns

df.shape

features = ['genres','keywords','original_language','title','cast','director']
for feature in features:
    df[feature] = df[feature].fillna('')

def combined_features(row):
    return row['title']+","+row['genres']+","+row['keywords']+","+row['original_language']+","+row['cast']+","+row['director']
df['combined_features'] = df.apply(combined_features,axis=1)
df['combined_features']

tfid = TfidfVectorizer()
tfidv = tfid.fit_transform(df['combined_features'])
tfidv.toarray()

tfidv.shape # vocabulary has 17502 words

cosine_sim = cosine_similarity(tfidv)
cosine_sim

cosine_sim.shape

movie = input("Enter Movie Name: ")
def get_index(mn):
    return df[df.title == mn].index[0] #returns index of the first matching title

mi = get_index(movie)
mi

sm = list(enumerate(cosine_sim[mi])) #enumerate: gives index to cosine similarity
print(sm) #gives similarity of mi with other movies

sorted_sm = sorted(sm,key=lambda x:x[1], reverse = True) #lambda x:x[1]:- want ot sort using cosine similarity and not index
print(sorted_sm)

def get_info(index):
    return df[df.index==index]['title'].values[0]+": "+df[df.index==index]['cast'].values[0]

i=0
for movie in sorted_sm:
    print(get_info(movie[0]))
    i=i+1
    if i>10:
        break









*********************            Bagging                ****************************8




from sklearn.datasets import load_iris
import pandas as pd

data=load_iris()
df=pd.DataFrame(data.data,columns=data.feature_names)
df['Species']=data.target
df

x=df.iloc[:,:-1]
y=df.iloc[:,-1]

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3,random_state=21)

from sklearn.tree import DecisionTreeClassifier
dtc=DecisionTreeClassifier()
dtc.fit(xtrain,ytrain)

predDTC=dtc.predict(xtest)

from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier()
rfc.fit(xtrain,ytrain)

predRFC=rfc.predict(xtest)

from sklearn.ensemble import BaggingClassifier
model=BaggingClassifier()
model.fit(xtrain,ytrain)

predBG=model.predict(xtest)

from sklearn.metrics import accuracy_score
print("Accuracy for Decision Tree=",accuracy_score(predDTC,ytest))
print("Accuracy for RandomForest=",accuracy_score(predRFC,ytest))
print("Accuracy for Bagging classifiers=",accuracy_score(predBG,ytest))





**********************        Boosting*                **************************

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

dataset=load_iris()
x=dataset.data
y=dataset.target
print(x.shape)
print(y.shape)

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3,random_state=1)

#AdaBoost

from sklearn.ensemble import AdaBoostClassifier
model=AdaBoostClassifier(n_estimators=50)
model.get_params

model.fit(xtrain,ytrain)

ypred=model.predict(xtest)

print(accuracy_score(ytest,ypred))

#Gradient Boosting

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score,classification_report
from sklearn.ensemble import GradientBoostingClassifier

train_data=pd.read_csv("train.csv")
test_data=pd.read_csv("test.csv")

ytrain=train_data["Survived"]
train_data.drop("Survived",axis=1,inplace=True)

from sklearn.preprocessing import LabelEncoder

lb=LabelEncoder()
lb.fit(train_data["Sex"])
train_data["Sex"]=lb.transform(train_data["Sex"])
train_data.fillna(value=0.0,inplace=True)

test_data["Sex"]=lb.transform(test_data["Sex"])
test_data.fillna(value=0.0,inplace=True)

drop_column=['Name',"Age",'SibSp','Parch',"Ticket","Cabin","Embarked"]
train_data.drop(labels=drop_column,axis=1,inplace=True)
test_data.drop(labels=drop_column,axis=1,inplace=True)

xtrain=train_data
xtest=test_data
print(xtrain.shape)
print(ytrain.shape)

scaler=MinMaxScaler()
xtrain=scaler.fit_transform(xtrain)
xtest=scaler.fit_transform(xtest)

xtrain,xval,ytrain,yval=train_test_split(xtrain,ytrain,test_size=0.3,random_state=2)

lr_list=[0.05,0.075,0.1,0.25,0.5,0.75,1]
for learning_rate in lr_list:
  gb_clf=GradientBoostingClassifier(n_estimators=20,learning_rate=learning_rate,
max_features=2,max_depth=2,random_state=0)
  gb_clf.fit(xtrain,ytrain)
  print(f"Learning Rate:{learning_rate:.3f}\tAccuracy Score(Training):{gb_clf.score(xtrain,ytrain):.3f}\tAccuracy Score(Testing):{gb_clf.score(xval,yval):.3f}")

gb_clf2=GradientBoostingClassifier(n_estimators=20,learning_rate=0.750,
max_features=2,max_depth=2,random_state=0)

gb_clf2.fit(xtrain,ytrain)

prediction=gb_clf2.predict(xval)

from sklearn.metrics import confusion_matrix,classification_report
print(classification_report(yval,prediction))
confusion_matrix(yval,prediction)





************************          STACKING             *********************************

from numpy import mean
from sklearn.datasets import make_regression #For making synthetic dataset
from sklearn.model_selection import cross_val_score, RepeatedKFold
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import StackingRegressor


x,y=make_regression(n_samples=1000,n_features=20,random_state=1)



def get_stacking():
    level10=list()
    level10.append(('knn',KNeighborsRegressor()))
    level10.append(('svm',SVR()))
    level1=LinearRegression()
    model=StackingRegressor(estimators=level10,final_estimator=level1)
    return model


def get_models():
    models=dict()
    models['knn']=KNeighborsRegressor()
    models['cart']=DecisionTreeRegressor()
    models['svm']=SVR()
    models['stacking']=get_stacking()
    return models


def evaluate_model(model,x,y):
    cv=RepeatedKFold(n_splits=10,n_repeats=3,random_state=1)
    scores=cross_val_score(model,x,y,scoring='neg_mean_absolute_error',cv=cv)
    return scores




models = get_models()
results,names=list(),list()
for name,model in models.items():
    scores=evaluate_model(model,x,y)
    results.append(scores)
    names.append(model)
    print(name,mean(scores))







************************         Genetic ALGO     *********************

pip install pygad

import pygad 
import numpy as np
x=[4,-2,3.5,5,-11,-4,7]
desired_output=44

def fitness_function(ga_instance,solution,solution_idx):
    output=np.sum(solution*x)
    fitness=1.0/np.abs(output-desired_output)
    return fitness

ga_instance=pygad.GA(num_generations=50,
                     crossover_type="single_point",
                     init_range_high=5,
                     init_range_low=-2,
                     num_genes=len(x),
                     fitness_func=fitness_function,
                     mutation_type="random",
                     mutation_percent_genes=10,
                     keep_parents=1,sol_per_pop=8,
                     parent_selection_type="sss",
                     num_parents_mating=8,)


ga_instance.run()

solution,solution_fitness,solution_idx=ga_instance.best_solution()

print("solution",solution)
print("fitness",solution_fitness)
print("Solution IDX",solution_idx)



*****************************         KNN using book reviews             ********************************

import pandas as pd

ratings = pd.read_csv("books_ratings.csv",,encoding = "latin-1",delimiter = ';')
ratings.head()

ratings_books = pd.merge(ratings,books,on='ISBN')

ratings_books_sample = ratings_books.sample(frac=.01,random_state=1)
ratings_books_sample.shape

ratings_books_pivot = ratings_books_sample.pivot_table(index = 'Book-Title',columns = 'Use
ratings_books_pivot.head()

from sklearn.neighbors import NearestNeighbors

model = NearestNeighbors(metric = 'cosine',algorithm='brute',n_neighbors=7,n_jobs=-1)

model.fit(ratings_books_pivot)

indices = model.kneighbors(ratings_books_pivot.loc[["01-01-00: The Novel of the Millennium
for index,value in enumerate(ratings_books_pivot.index[indices][0]):
    print((index+1),".",value)












