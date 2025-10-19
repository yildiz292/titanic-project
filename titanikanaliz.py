import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


data=sns.load_dataset("titanic")
df=data.drop(["deck","embark_town","who","adult_male"],axis=1)
#Here we delete unnecessary lines,this makes the data more undersdanble and reduces the possibilty of making errors

df["age"].fillna(df["age"].median(),inplace=True)
df["embarked"].fillna(df["embarked"].mode()[0],inplace=True)
#Missing values were filled to improve model accuracy.The missing age values were replaced with the mean age,
# and missing embarked values were filled with the most frequent port.

df['sex'] = df['sex'].map({'male':0,'female':1})
df['embarked'] = df['embarked'].map({'S':0,'C':1,'Q': 2})
#The sex column was converted to numeric values for model training.Females were encoded as 0 and males as 1.
# This allows the model to learn any relaationship between gender an survival.

df=pd.get_dummies(df,columns=["class","alive","alone"],drop_first=True)
#I used the get_dummies() method to convert categorical veriables into numercial format for machine learning.
x=df[['pclass','sex','age','fare','sibsp','parch','embarked']]
y=df["survived"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model=LogisticRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print("Accuracy:",accuracy_score(y_test,y_pred) * 100)
#The accuracy of the model was calculated using the 'accuracy_score' metric.
# This shows the proportion of correct predictions made by the model.
for cls in sorted(x_test['pclass'].unique()):
    cls_pred=y_pred[x_test['pclass']== cls]
    print(f"pclass {cls} tahmin ortalaması:",cls_pred.mean())
#Here it shows the survival rates for that class by averaging.
yeniyolcu=np.array([[1,1,25,100,0,0,0]])
tahmin=model.predict(yeniyolcu)
print("new passenger",tahmin[0])
#Here we can find  whether a new passenger has survived or not by adding his/her features.
young_pred=y_pred[x_test['age']< 30]
old_pred=y_pred[x_test['age'] >= 30]
print("Average prediction for those under 30:",young_pred.mean())
print("Average prediction for ages 30 and above:",old_pred.mean())
#Estimates mean age.
ortalamayaş=df["age"].mean()
print(ortalamayaş)
#Generally,we see the average age,that is which age is the highest.
#Here is a pie chart comparing the number of people who died and survived.
hayatakalma=df["survived"].value_counts()
labels=['those who die','survivos']
colors=["lightblue",'blue']
plt.pie(hayatakalma,labels=labels,autopct='%1.1f%%',colors=colors,shadow=True,startangle=90)
plt.title("")
plt.show()
#Here is a pie chart comparing the number of people who died and survived and as can be seen,
# there are more people who died than there are people who survived.
embarked_survived=data.groupby("embarked")["survived"].sum()
embarked_total=data["embarked"].value_counts()
embarkedd=pd.DataFrame({"total passengers":embarked_total,"Survived":embarked_survived})
embarkedd.plot(kind="bar")
plt.title("Survival by emvarked")
plt.show()
#Here is a graph of survival rates by port type
survivedandsex=data.groupby("sex")["survived"].sum()
plt.pie(survivedandsex,labels=survivedandsex.index,autopct='%1.1f%%',shadow=True)
plt.title("Survival by gender")
plt.show()
#Survival chart by gender
sns.barplot(data=df,x="pclass",y="survived",color="pink")
plt.title("Survival rate by ticket")
plt.show()
#Survival by pclass
sns.histplot(df["age"],bins=30,kde=True)
plt.title("Age")
plt.show()
#Age chart
alone_count=df[(df['sibsp'] + df['parch']) == 0].shape[0]
print(alone_count)
sns.catplot(data=data,x="survived",y="alone_count")
plt.title("")
plt.show()
#How many people traveled alone





