# import pandas as pd
# from sklearn.preprocessing import LabelEncoder
# from sklearn.naive_bayes import CategoricalNB

# data = {
#     'Age': ['Young','Young','Middle','Old','Old','Middle'],
#     'Obesity': ['Yes','No','Yes','Yes','No','No'],
#     'BP': ['High','Normal','High','Normal','Normal','High'],
#     'Diabetic': ['Yes','No','Yes','Yes','No','No']
# }
# df = pd.DataFrame(data)

# X = df[['Age', 'Obesity', 'BP']].copy()
# y = df['Diabetic'].copy()


# le_age = LabelEncoder()
# le_obesity = LabelEncoder()
# le_bp = LabelEncoder()
# le_target = LabelEncoder()


# X['Age'] = le_age.fit_transform(X['Age'])
# X['Obesity'] = le_obesity.fit_transform(X['Obesity'])
# X['BP'] = le_bp.fit_transform(X['BP'])
# y = le_target.fit_transform(y)

# model = CategoricalNB()
# model.fit(X, y)

# test = pd.DataFrame({
#     'Age': ['Young'],
#     'Obesity': ['Yes'],
#     'BP': ['Normal']
# })
# test['Age'] = le_age.transform(test['Age'])
# test['Obesity'] = le_obesity.transform(test['Obesity'])
# test['BP'] = le_bp.transform(test['BP'])


# print(X)
# print(y)
# pred = model.predict(test)
# print(le_target.inverse_transform(pred)[0]) 



# q2


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score

data = {
    'Height': [170,160,175,165,180,158],
    'Weight': [65,55,70,60,75,50],
    'HairLength': [5,20,4,18,6,22],
    'Gender': ['Male','Female','Male','Female','Male','Female']
}
df = pd.DataFrame(data)

X = df[['Height','Weight','HairLength']].copy()
y = df['Gender'].copy()

le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))

test = pd.DataFrame({'Height':[168],'Weight':[62],'HairLength':[15]})
pred = model.predict(test)
print("Predicted Gender:", le.inverse_transform(pred)[0])