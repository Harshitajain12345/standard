import numpy as np
import pandas as pd
df = pd.read_csv('covid.csv')
df.head(2)
df = df.dropna()
from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
df['gender'] = lb.fit_transform(df['gender'])
df['has_covid'] = lb.fit_transform(df['has_covid'])
df['cough'] = lb.fit_transform(df['cough'])
df['city'] = lb.fit_transform(df['city'])
df.head(3)
x=df.drop(columns = ['has_covid'])
y = df['has_covid']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 42)
np.round(x_train.describe() , 2)
from sklearn.preprocessing import MinMaxScaler
mn = MinMaxScaler()
x_train_mn = mn.fit_transform(x_train)
x_train_new = pd.DataFrame(x_train_mn , columns = x_train.columns)
np.round(x_train_new.describe() , 2)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train_sc = sc.fit_transform(x_train)
x_train_new = pd.DataFrame(x_train_sc , columns = x_train.columns)
x_train_new.head(2)
np.round(x_train_new.describe() , 2)