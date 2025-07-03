import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.decomposition import PCA


''' Loading dataset '''

data = pd.read_csv('C:/Desktop/Elevate Labs/Task 7/breast-cancer.csv')
data.drop('id', axis=1, inplace=True)
data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})


''' Features & Target '''

X = data.drop('diagnosis', axis=1)
y = data['diagnosis']


''' Train-Test Split '''

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


''' Feature Scaling '''

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


''' 1 Linear Kernel SVM '''

model_linear = SVC(kernel='linear', C=1)
model_linear.fit(X_train, y_train)
pred_linear = model_linear.predict(X_test)

print("Linear SVM Accuracy:", accuracy_score(y_test, pred_linear))


''' 2 RBF Kernel SVM '''

model_rbf = SVC(kernel='rbf', C=1, gamma='scale')
model_rbf.fit(X_train, y_train)
pred_rbf = model_rbf.predict(X_test)

print("RBF Kernel SVM Accuracy:", accuracy_score(y_test, pred_rbf))


''' 3 Visualize with PCA (2D Plot) '''

pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

X_train2, X_test2, y_train2, y_test2 = train_test_split(X_reduced, y, test_size=0.2, random_state=42)

model_vis = SVC(kernel='linear', C=1)
model_vis.fit(X_train2, y_train2)


''' Plot Decision Boundary '''

x_min, x_max = X_reduced[:, 0].min() - 1, X_reduced[:, 0].max() + 1
y_min, y_max = X_reduced[:, 1].min() - 1, X_reduced[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.5),
                     np.arange(y_min, y_max, 0.5))  # Increased step size

Z = model_vis.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.contourf(xx, yy, Z, cmap='coolwarm', alpha=0.5)
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap='coolwarm', edgecolor='k')
plt.title('Decision Boundary with Linear SVM')
plt.show()


''' 4 Hyperparameter Tuning (Grid Search) '''
param_grid = {'C': [0.1, 1, 10], 'gamma': ['scale', 0.01, 0.1], 'kernel': ['rbf']}
grid = GridSearchCV(SVC(), param_grid, cv=3)
grid.fit(X_train, y_train)

print("Best Parameters:", grid.best_params_)
print("Best Accuracy after tuning:", grid.best_score_)
