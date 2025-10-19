import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data=pd.read_csv("train.csv")
#dat=pd.read_csv("test.csv")

#df =sns.load_dataset('titanic')
data['AgeGroup'] =pd.cut(data['Age'],bins=[0,12,18,30,50,80],labels=["bebe","velet","ergen","yetiskin","orta","yaslı"])
sns.barplot(x='AgeGroup',y='Survived',data=data)
plt.show()





