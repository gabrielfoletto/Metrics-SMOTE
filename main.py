from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

#Load data from CSV file
dataset_path = "Dataset.csv"
data = pd.read_csv(dataset_path, sep=";")

#Map label classes to integer numbers
label_mapping = {
    'Normal': 0,
    'Conti': 1,
    'LockBit': 2,
    'Ryuk': 3,
    'Sodinikibi': 4,
    'WannaCry': 5,
    'CryptoLocker': 6
}

data['classe'] = data['classe'].map(label_mapping)

#Separate features and label
X = data.iloc[:, :-1]
y = data['classe']

#Apply SMOTE for class balancing
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

#Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

#Train the XGBoost model
model = xgb.XGBClassifier()
model.fit(X_train, y_train)

#Make predictions on the test set
y_pred = model.predict(X_test)

#Calculate metrics for the entire dataset
report_overall = precision_recall_fscore_support(y_test, y_pred, average='weighted')
accuracy_overall = accuracy_score(y_test, y_pred)

#Save metrics to a CSV file
df_overall = pd.DataFrame({
    'Metric': ['Precision', 'Recall', 'F1-Score', 'Accuracy'],
    'Score': [report_overall[0], report_overall[1], report_overall[2], accuracy_overall]
})
df_overall.to_csv('metrics_dataset.csv', index=False)

#Calculate metrics for each class
for classe in label_mapping:
    mask = (y_test == label_mapping[classe])
    y_true_class = y_test[mask]
    y_pred_class = y_pred[mask]
    
    if len(y_true_class) > 0:  #Check if there are samples for this class
        metrics_class = precision_recall_fscore_support(y_true_class, y_pred_class, average='weighted')
        precision_class = metrics_class[0]
        recall = metrics_class[1]
        f1_score = metrics_class[2]
        accuracy = accuracy_score(y_true_class, y_pred_class)
    else:
        precision_class = recall = f1_score = accuracy = 0.0
    
    #Save metrics to CSV files
    df_class = pd.DataFrame({
        'Metric': ['Precision', 'Recall', 'F1-Score', 'Accuracy'],
        'Score': [precision_class, recall, f1_score, accuracy]
    })
    df_class.to_csv(f'metrics_{classe}.csv', index=False)
