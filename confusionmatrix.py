import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import seaborn as sns
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score

y_act=[1,1,1,0,0,1,0,1,0]
y_pred = [1,1,0,1,1,0,1,1,0]

cm=confusion_matrix(y_act,y_pred)

tn,fp,fn,tp = cm.ravel()        #this is standard sequence

print("Confusion_matrix \n", cm )

print(f"True Positive={tp}, False Positive={fp},False Negative={fn},True Negative = {tn}")

print("Accuracy",accuracy_score(y_act,y_pred))
print("Precision", precision_score(y_act,y_pred))
print("Recall", recall_score(y_act,y_pred))
print("F1 Score",f1_score(y_act,y_pred))
print("Specificitty",tn/(tn+fp))

plt.figure(figsize=(5,6))

sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",xticklabels=["Predected True","Predicted False"],yticklabels=["Actual True","Actual False"])

plt.title("Actual vs Predicted")

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


