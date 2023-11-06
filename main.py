from ucimlrepo import fetch_ucirepo
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from aco import FeatureSelectionACO
from analyzer import DataAnalyzer
from sklearn.pipeline import Pipeline


def load_data(id=17):
    data = fetch_ucirepo(id=id)
    X = data.data.features
    y = data.data.targets
    return X, y


def load_data(id=17):
    data = fetch_ucirepo(id=id)
    X = data.data.features
    y = data.data.targets
    return X, y

X, y = load_data()

# analyzer = DataAnalyzer(X, y)
#
# analyzer.correlation_matrix()
# analyzer.perform_univariate_analysis()
# analyzer.analyze_columns()
#
clf = SVC(kernel='linear', C=1, probability=True)
num_ants = 10
num_iterations = 50

model = Pipeline(steps=[
    ('scaler', MinMaxScaler()),
    ('aco_featureselection', FeatureSelectionACO(clf, X.shape[1], num_ants, num_iterations)),
    ('clf_SVC', clf)
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# treinando o modelo
model.fit(X_train, np.ravel(y_train))

train_score = model.score(X_train, y_train)

# avaliando o modelo
test_score = model.score(X_test, y_test)

print("Train score: {}".format(train_score))
print("Test score: {}".format(test_score))

'''
# Initialize classifiers
random_forest_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
svm_classifier = SVC(kernel='linear', C=1, probability=True)

# Train classifiers
random_forest_classifier.fit(X_train, y_train)
svm_classifier.fit(X_train, y_train)

# Predictions
rf_predictions = random_forest_classifier.predict(X_test)
svm_predictions = svm_classifier.predict(X_test)

# Confusion matrices and classification reports
print("Random Forest Classifier:")
print("Confusion Matrix:\n", confusion_matrix(y_test, rf_predictions))
print("\nClassification Report:\n", classification_report(y_test, rf_predictions))

print("\nSVM Classifier:")
print("Confusion Matrix:\n", confusion_matrix(y_test, svm_predictions))
print("\nClassification Report:\n", classification_report(y_test, svm_predictions))

# Feature Selections

num_ants = 15
num_iterations = 70

clf = RandomForestClassifier(n_estimators=100, random_state=42)
selected_features, iterations, accuracies = FeatureSelectionACO(clf, X_train,
                                                                y_train, X_test,
                                                                y_test, num_ants,
                                                                num_iterations)

# Create a progress plot
plt.figure(figsize=(10, 6))
plt.plot(iterations, accuracies, marker='o', linestyle='-', color='b')
plt.title("ACO Feature Selection Progress")
plt.xlabel("Iteration")
plt.ylabel("Best Accuracy")
plt.grid(True)
plt.show()

# Use the selected features for further analysis
X_train_selected = X_train.iloc[:, selected_features].values
X_test_selected = X_test.iloc[:, selected_features].values

# Train and evaluate a classifier with the selected features using SVC
clf_selected = SVC(kernel='linear', random_state=42)
clf_selected.fit(X_train_selected, y_train)
y_pred_selected = clf_selected.predict(X_test_selected)
accuracy = accuracy_score(y_test, y_pred_selected)
print("Accuracy with Selected Features:", accuracy)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_selected))
print("\nClassification Report:\n", classification_report(y_test, y_pred_selected))
print(X.iloc[:, selected_features].columns)
'''