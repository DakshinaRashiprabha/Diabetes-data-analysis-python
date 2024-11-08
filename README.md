import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics as sm
matplotlib inline
diabetes_dataset = pd.read_csv('diabetes.csv')


diabetes_dataset
![image](https://github.com/user-attachments/assets/faa08f95-27fc-4ca9-9399-c27fba986c30)

diabetes_dataset.describe()
![image](https://github.com/user-attachments/assets/e3a4c522-4c53-4ab1-b9ac-fc410fd76124)


sum(diabetes_dataset.isnull().sum())
![image](https://github.com/user-attachments/assets/b57c5d72-a884-4234-a094-59e8233522e5)


print((diabetes_dataset[['Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']]==0).sum())
![image](https://github.com/user-attachments/assets/e2c4e091-8f5d-4c92-bad7-2871f5bded16)


diabetes_dataset[['Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']] = diabetes_dataset[['Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']].replace(0,np.NAN)
![image](https://github.com/user-attachments/assets/5c594232-feee-49e9-9371-054023b6fe94)

diabetes_dataset.fillna(diabetes_dataset.mean(), inplace=True)
print(diabetes_dataset.isnull().sum())
![image](https://github.com/user-attachments/assets/036aa3dc-6510-47c3-ae17-5f653ace54b8)



diabetes_dataset
![image](https://github.com/user-attachments/assets/d01140a5-c09a-4e77-9a11-086097192f1d)


from sklearn.preprocessing import LabelEncoder

LabelEncoder = LabelEncoder()

dataTransform = diabetes_dataset.copy()

for data in diabetes_dataset.columns:
    dataTransform[data] = LabelEncoder.fit_transform(diabetes_dataset[data])
    
dataTransform
![image](https://github.com/user-attachments/assets/e9ce7c3a-356e-4ba1-bc24-dea2209b181c)


x = dataTransform.drop(['Outcome'], axis=1)
x
![image](https://github.com/user-attachments/assets/36ccfa91-a1e2-46ab-9e90-9cba76c206a6)

y = dataTransform['Outcome']
y
![image](https://github.com/user-attachments/assets/848e7f08-19ee-442d-a791-ada0fb38cd97)

diabetes_features_list = list(x.columns)

from sklearn.model_selection import train_test_split

x_train ,x_test,y_train,y_test = train_test_split (x,y,test_size = 0.2,random_state = 41)

from sklearn.ensemble import RandomForestClassifier

randomforestclassifier = RandomForestClassifier(n_estimators = 1200)

prediction_y = randomforestclassifier.predict(x_test)

prediction_y
![image](https://github.com/user-attachments/assets/b8fe1076-d0d9-4c31-8ede-22ecac3fcfde)

experiment_accuracy = sm.accuracy_score(y_test,prediction_y)
print ('Accuracy Score is :' , str(experiment_accuracy))

Accuracy Score is : 0.995

from sklearn import metrics
print("Classification Report :",metrics.classification_report(prediction_y,y_test,target_names=["Diabetes","No Diabetes"]))
![image](https://github.com/user-attachments/assets/01f4db55-37bb-4a3a-8752-4a31261f4fb5)

from sklearn.metrics import confusion_matrix

import seaborn as sb

sb.set()

get_ipython().run_line_magic('matplotlib','inline')

import matplotlib.pyplot as pt

confusionmt = confusion_matrix(y_test,prediction_y)

sb.heatmap(confusionmt.T, square=True, annot=True,fmt='d', cbar=False)
pt.xlabel('true class axis')
pt.ylabel('predicted class axis')
![image](https://github.com/user-attachments/assets/3a797ed4-4d9d-4775-8bc7-98c06783d85f)

