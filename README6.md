Support Vector Machines (SVM) - Breast Cancer Dataset

Objective:
Implement and understand SVM for binary classification, evaluate model performance, experiment with linear and RBF kernels, visualize decision boundary, and tune hyperparameters.

Tools & Libraries:
- Python
- pandas, numpy
- scikit-learn
- matplotlib

Dataset:
Filename: breast-cancer.csv
Description: Breast Cancer dataset with tumor features and diagnosis labels.

Features Include:
- Radius, Texture, Perimeter, Area, Smoothness, etc.
- Target Variable: diagnosis (Malignant = 1, Benign = 0)

Steps Performed:
✔️ Data Preprocessing and Encoding of Target
✔️ Feature Normalization using StandardScaler
✔️ Trained SVM Classifier with Linear Kernel
✔️ Trained SVM Classifier with RBF Kernel
✔️ Visualized Decision Boundary using PCA (2D)
✔️ Tuned Hyperparameters (C, gamma) using GridSearchCV
✔️ Evaluated Models using Accuracy and Confusion Matrix

Results Summary:
- Linear SVM model tested on Breast Cancer dataset
- RBF SVM model tested with gamma parameter tuning
- Best parameters selected through GridSearchCV
- Achieved high classification accuracy
- Decision Boundary visualized for understanding separation

How to Run:
1. Install required libraries:
   pip install pandas numpy scikit-learn matplotlib

2. Place 'breast-cancer.csv' dataset in the same folder as the Python script.

3. Run the Python script:
   python Task7.py

Key Learnings:
- SVM maximizes margin for better classification
- Kernel trick enables handling non-linear problems
- Proper tuning of C and gamma controls overfitting
- PCA helps visualize complex decision boundaries

Submission Notes:
- Included clean Python code file
- Dataset used: breast-cancer.csv
- Visualizations generated during runtime
- README provided explaining task workflow
