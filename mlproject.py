# Loan_Prediction_Project.ipynb  # <- Notebook title comment
# Requirements: pandas, numpy, scikit-learn, matplotlib, seaborn, graphviz, pydotplus  # <- Packages needed

import pandas as pd  # <- Import pandas for data handling
import numpy as np   # <- Import numpy for numerical operations
import matplotlib.pyplot as plt  # <- For plotting charts
import seaborn as sns  # <- For advanced visualizations

# Machine learning utilities from sklearn
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score  # <- For splitting & tuning
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler  # <- Data preprocessing tools
from sklearn.impute import SimpleImputer  # <- To fill missing values
from sklearn.pipeline import Pipeline  # <- To chain steps together
from sklearn.compose import ColumnTransformer  # <- To apply transforms to specific columns

# Classification algorithms
from sklearn.tree import DecisionTreeClassifier, plot_tree  # <- Decision tree + visualization
from sklearn.ensemble import RandomForestClassifier  # <- Ensemble Random Forest model
from sklearn.linear_model import LogisticRegression  # <- Logistic Regression model
from sklearn.svm import SVC  # <- Support Vector Machine
from sklearn.neighbors import KNeighborsClassifier  # <- K-Nearest Neighbors

# Evaluation metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,  # <- Main accuracy metrics
    confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc,  # <- Confusion matrix + ROC
    classification_report  # <- Detailed precision/recall report
)

# -----------------------------------------------------------
# 1. Load dataset
# -----------------------------------------------------------

df = pd.read_csv('train.csv')  # <- Load CSV dataset into pandas dataframe
print("Shape:", df.shape)  # <- Print number of rows and columns
df.head()  # <- Show first 5 rows of dataset

# -----------------------------------------------------------
# 2. Quick EDA
# -----------------------------------------------------------

print(df.info())  # <- Show data types + missing values summary
print(df.isnull().sum())  # <- Print count of missing values column-wise
print(df.describe(include='all'))  # <- Statistical summary for all columns

# -----------------------------------------------------------
# 3. Preprocessing plan
# -----------------------------------------------------------

target = 'Loan_Status'  # <- Target variable column name
X = df.drop(columns=[target, 'Loan_ID'], errors='ignore')  # <- Drop ID + target from features
y = df[target].map({'Y': 1, 'N': 0})  # <- Convert 'Y'/'N' labels to 1/0

cat_cols = X.select_dtypes(include=['object']).columns.tolist()  # <- Get all categorical columns
num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()  # <- Get all numeric columns

print("Categorical:", cat_cols)  # <- Display categorical column names
print("Numeric:", num_cols)  # <- Display numeric column names

# -----------------------------------------------------------
# 4. Imputation + Encoding
# -----------------------------------------------------------

num_transformer = Pipeline(steps=[  # <- Pipeline for numeric columns
    ('imputer', SimpleImputer(strategy='median')),  # <- Fill missing numbers with median
    ('scaler', StandardScaler())  # <- Standardize numeric values
])

cat_transformer = Pipeline(steps=[  # <- Pipeline for categorical columns
    ('imputer', SimpleImputer(strategy='most_frequent')),  # <- Fill missing with most common value
    ('label_enc', OneHotEncoder(handle_unknown='ignore', sparse_output=False))  # <- Convert categories to 0/1
])

preprocessor = ColumnTransformer(transformers=[  # <- Combine num + cat preprocessors
    ('num', num_transformer, num_cols),  # <- Numeric transformer
    ('cat', cat_transformer, cat_cols)   # <- Categorical transformer
])

# -----------------------------------------------------------
# 5. Train-test split
# -----------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(  # <- Split into training & testing data
    X, y, test_size=0.2, random_state=42, stratify=y  # <- 20% test + maintain class balance
)

# -----------------------------------------------------------
# 6. Model training helper
# -----------------------------------------------------------

def train_and_eval(model, model_name):  # <- Function to train and evaluate any model
    pipe = Pipeline(steps=[  # <- Build pipeline (preprocess + model)
        ('preprocessor', preprocessor),  # <- Preprocess data
        ('classifier', model)  # <- Run classifier
    ])

    pipe.fit(X_train, y_train)  # <- Train the model on training data
    y_pred = pipe.predict(X_test)  # <- Predict on test data

    y_proba = None  # <- Initialize probabilities as None
    if hasattr(pipe, "predict_proba"):  # <- Check if model supports probability
        try:
            y_proba = pipe.predict_proba(X_test)[:, 1]  # <- Take probability of class 1
        except:
            y_proba = None  # <- Fail safe

    acc = accuracy_score(y_test, y_pred)  # <- Compute accuracy
    prec = precision_score(y_test, y_pred, zero_division=0)  # <- Precision
    rec = recall_score(y_test, y_pred, zero_division=0)  # <- Recall
    f1 = f1_score(y_test, y_pred, zero_division=0)  # <- F1 Score

    print(f"--- {model_name} ---")  # <- Print model name
    print("Accuracy:", round(acc, 4))  # <- Print accuracy
    print("Precision:", round(prec, 4))  # <- Print precision
    print("Recall:", round(rec, 4))  # <- Print recall
    print("F1-score:", round(f1, 4))  # <- Print F1 score
    print("Confusion Matrix:")  # <- Print confusion matrix title
    print(confusion_matrix(y_test, y_pred))  # <- Display confusion matrix
    print(classification_report(y_test, y_pred, digits=4))  # <- Detailed metrics

    if y_proba is not None:  # <- If model gives probability
        fpr, tpr, _ = roc_curve(y_test, y_proba)  # <- Compute ROC curve
        roc_auc = auc(fpr, tpr)  # <- Compute AUC score
        print("AUC:", round(roc_auc, 4))  # <- Print AUC
        return pipe, (fpr, tpr, roc_auc)  # <- Return model + ROC data

    return pipe, None  # <- Return model if no ROC available

# -----------------------------------------------------------
# 7. Fit models
# -----------------------------------------------------------

models = {  # <- Dictionary of ML models to train
    "DecisionTree": DecisionTreeClassifier(random_state=42, max_depth=4),  # <- Decision tree
    "RandomForest": RandomForestClassifier(random_state=42, n_estimators=100),  # <- Random forest
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),  # <- Logistic regression
    "SVM": SVC(kernel='rbf', probability=True, random_state=42),  # <- Support vector machine
    "KNN": KNeighborsClassifier(n_neighbors=5)  # <- K nearest neighbors
}

results = {}  # <- Store trained models
rocs = {}  # <- Store ROC curve data

for name, mdl in models.items():  # <- Loop through all models
    pipe, roc_data = train_and_eval(mdl, name)  # <- Train and evaluate model
    results[name] = pipe  # <- Save model pipeline
    if roc_data:  # <- If ROC exists
        rocs[name] = roc_data  # <- Save ROC data

# -----------------------------------------------------------
# 8. Plot ROC curves
# -----------------------------------------------------------

plt.figure(figsize=(8, 6))  # <- Initialize plot size
for name, (fpr, tpr, roc_auc) in rocs.items():  # <- Loop through ROC data
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')  # <- Plot ROC curve

plt.plot([0, 1], [0, 1], 'k--')  # <- Reference line (random guess)
plt.xlabel('False Positive Rate')  # <- X-axis label
plt.ylabel('True Positive Rate')  # <- Y-axis label
plt.title('ROC Curves')  # <- Plot title
plt.legend()  # <- Show legend
plt.show()  # <- Display plot

# -----------------------------------------------------------
# 9. Feature importance (Random Forest)
# -----------------------------------------------------------

ohe = results['DecisionTree'].named_steps['preprocessor']\
    .named_transformers_['cat'].named_steps['label_enc']  # <- Extract fitted OneHotEncoder

num_features = num_cols  # <- Store numeric feature names
cat_features = list(ohe.get_feature_names_out(cat_cols))  # <- Store encoded categorical names

features = num_features + cat_features  # <- Combine all features

rf = results['RandomForest'].named_steps['classifier']  # <- Get Random Forest model
importances = rf.feature_importances_  # <- Get feature importances

imp_df = pd.DataFrame({  # <- Build dataframe for plot
    'feature': features,  # <- Column of feature names
    'importance': importances  # <- Column of importances
}).sort_values(by='importance', ascending=False).head(20)  # <- Top 20 features

plt.figure(figsize=(8, 6))  # <- Set figure size
sns.barplot(x='importance', y='feature', data=imp_df)  # <- Plot bar chart
plt.title('Top 20 Feature Importances (RandomForest)')  # <- Title
plt.tight_layout()  # <- Adjust layout
plt.show()  # <- Show plot

# -----------------------------------------------------------
# 10. Decision Tree visualization
# -----------------------------------------------------------

dt = results['DecisionTree'].named_steps['classifier']  # <- Get trained decision tree

X_train_trans = results['DecisionTree'].named_steps['preprocessor'].transform(X_train)  # <- Transform training data

fitted_pre = preprocessor.fit(X_train)  # <- Fit preprocessor to extract names
num_features = num_cols  # <- Numeric feature names

try:
    ohe = fitted_pre.named_transformers_['cat'].named_steps['label_enc']  # <- Extract OHE
    cat_features = list(ohe.get_feature_names_out(cat_cols))  # <- Get encoded feature names
except:
    cat_features = []  # <- If error, leave empty

features = num_features + cat_features  # <- Combine all names

plt.figure(figsize=(22, 10))  # <- Large figure for readability
plot_tree(
    dt,  # <- Decision Tree classifier
    filled=True,  # <- Color nodes
    feature_names=features,  # <- Names of input features
    class_names=['Not Approved', 'Approved'],  # <- Labels for target classes
    max_depth=4,  # <- Limit tree depth
    fontsize=8  # <- Reduce font size
)
plt.title("Decision Tree Visualization (Loan Approval Prediction)")  # <- Plot title
plt.show()  # <- Display visualization
