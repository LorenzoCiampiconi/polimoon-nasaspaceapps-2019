import numpy as np # linear algebra
import pandas as pd # data processing
import warnings
import datetime
import matplotlib.pyplot as plt    # basic plotting library
import seaborn as sns              # more advanced visual plotting library
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
warnings.filterwarnings("ignore")

DataFrame = pd.read_csv("/home/ago/Scrivania/NASA_2019/lunar_nan_dropped.csv")
DataFrame_filled = pd.read_csv("/home/ago/Scrivania/NASA_2019/NASA_CLASSIFIER/VER2/Luna_Geocentric_Clean.csv")

#drop date
#DataFrame.drop(columns=['Date'],inplace=True)



DataFrame['Date'] = (pd.to_datetime(DataFrame['Date'])).dt.month



DataFrame.info()   # information about data types and amount of non-null rows of our Dataset
print(DataFrame.describe())   # statistical information about our data
print(DataFrame.corr())    # correlation between fields


#PRE PROCESSING

#splitting Feature and Label
labels = DataFrame.Constellation.values
DataFrame.drop(["Constellation"],axis=1,inplace=True)
features = DataFrame.values

#scaling 
scaler = MinMaxScaler(feature_range=(0,1))
features_scaled = scaler.fit_transform(features)

#splitting the train and the test rows
x_train, x_test, y_train, y_test = train_test_split(features_scaled,labels,test_size=0.1)

#randomForestClassifier
rfc_model = RandomForestClassifier(n_estimators=237,random_state=42,max_leaf_nodes=300,criterion="entropy")
rfc_model.fit(x_train,y_train)
y_head_rfc = rfc_model.predict(x_test)
rfc_score = rfc_model.score(x_test,y_test)

#confusionMatrix
cm_rfc = confusion_matrix(y_test,y_head_rfc)

# Train and Test Accuracy
print ("Train Accuracy :: ", accuracy_score(y_train,rfc_model.predict(x_train)))
print ("Test Accuracy  :: ", accuracy_score(y_test, y_head_rfc))

plt.title("Random Forest Confusion Matrix")
sns.heatmap(cm_rfc,cbar=False,annot=True,cmap="CMRmap_r",fmt="d")
plt.show()