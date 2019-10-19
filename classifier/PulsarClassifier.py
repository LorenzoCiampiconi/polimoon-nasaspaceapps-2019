import numpy as np # linear algebra
import pandas as pd # data processing
import warnings
import matplotlib.pyplot as plt    # basic plotting library
import seaborn as sns              # more advanced visual plotting library
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
warnings.filterwarnings("ignore")

DataFrame = pd.read_csv("pulsar_stars.csv")
print(DataFrame.head())    # first 5 rows of whole columns
DataFrame.info()   # information about data types and amount of non-null rows of our Dataset
print(DataFrame.describe())   # statistical information about our data
print(DataFrame.corr())    # correlation between fields

sns.pairplot(data=DataFrame,
             palette="husl",
             hue="target_class",
             vars=[" Mean of the integrated profile",
                   " Excess kurtosis of the integrated profile",
                   " Skewness of the integrated profile",
                   " Mean of the DM-SNR curve",
                   " Excess kurtosis of the DM-SNR curve",
                   " Skewness of the DM-SNR curve"])

plt.suptitle("PairPlot of Data Without Std. Dev. Fields",fontsize=18)

plt.tight_layout()
plt.show()   # pairplot without standard deviaton fields of data


#PRE PROCESSING

#splitting Feature and Label
labels = DataFrame.target_class.values
DataFrame.drop(["target_class"],axis=1,inplace=True)
features = DataFrame.values

#scaling 
scaler = MinMaxScaler(feature_range=(0,1))
features_scaled = scaler.fit_transform(features)

#splitting the train and the test rows
x_train, x_test, y_train, y_test = train_test_split(features_scaled,labels,test_size=0.2)

#randomForestClassifier
rfc_model = RandomForestClassifier(n_estimators=37,random_state=42,max_leaf_nodes=200,criterion="entropy")
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