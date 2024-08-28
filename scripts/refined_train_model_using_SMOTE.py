import os
import time
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.combine import SMOTEENN
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from joblib import dump
import matplotlib.pyplot as plt
import logging
from category_encoders import TargetEncoder
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def save_confusion_matrix(cm, model_name, save_dir):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix for {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()

    # Define the directory path for saving the confusion matrix
    os.makedirs(save_dir, exist_ok=True)

    plt.savefig(os.path.join(save_dir, f'{model_name.lower().replace(" ", "_")}_confusion_matrix.png'))
    plt.close()


def evaluate_model(model, X_test, y_test,model_name):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # Save the confusion matrix
    save_dir = os.path.join(ROOT_DIR, 'confusion_matrices')
    save_confusion_matrix(cm, model_name, save_dir)

    return accuracy, precision, recall, f1, roc_auc, cm


def train_models(X_train, X_test, y_train, y_test):
    logger.info("Fitting and transforming the training data")

    # Drop the 'url' column if exists
    if 'url' in X_train.columns:
        X_train = X_train.drop(columns=['url'])
        X_test = X_test.drop(columns=['url'])

    # Separate numeric and non-numeric features
    numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X_train.select_dtypes(include=['object']).columns

    imputer = SimpleImputer(strategy='mean')
    X_train_numeric = imputer.fit_transform(X_train[numeric_features])
    X_test_numeric = imputer.transform(X_test[numeric_features])

    scaler = StandardScaler()
    X_train_numeric = scaler.fit_transform(X_train_numeric)
    X_test_numeric = scaler.transform(X_test_numeric)

    # Use target encoding for categorical features
    target_encoder = TargetEncoder(cols=categorical_features)
    X_train_categorical = target_encoder.fit_transform(X_train[categorical_features], y_train)
    X_test_categorical = target_encoder.transform(X_test[categorical_features])

    # Recombine numeric and categorical features
    X_train_processed = pd.DataFrame(X_train_numeric, columns=numeric_features)
    X_test_processed = pd.DataFrame(X_test_numeric, columns=numeric_features)

    X_train_processed = pd.concat([X_train_processed, X_train_categorical.reset_index(drop=True)], axis=1)
    X_test_processed = pd.concat([X_test_processed, X_test_categorical.reset_index(drop=True)], axis=1)

    logger.info("Applying SMOTEENN to the training data")
    start_time = time.time()

    # Combine SMOTE and ENN (Edited Nearest Neighbors) to balance the dataset
    smote_enn = SMOTEENN(smote=SMOTE(random_state=42), random_state=42)
    smote_enn = SMOTEENN(smote=SMOTE(random_state=42), random_state=42)
    X_train_resampled, y_train_resampled = smote_enn.fit_resample(X_train_processed, y_train)
    logger.info(f"SMOTEENN applied in {time.time() - start_time:.2f} seconds")

    # Print the count of malicious and benign URLs in the resampled dataset
    unique, counts = np.unique(y_train_resampled, return_counts=True)
    resampled_counts = dict(zip(unique, counts))
    logger.info(f"Resampled dataset counts: {resampled_counts}")

    os.makedirs(os.path.join(ROOT_DIR, 'models'), exist_ok=True)

    models = {
        "Random Forest": RandomForestClassifier(random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Linear SVM": LinearSVC(random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "KNN": KNeighborsClassifier(),
        "XGBoost": XGBClassifier(random_state=42),
        "LightGBM": LGBMClassifier(random_state=42, verbosity=-1)
    }

    results = []
    feature_importances = {}

    for name, model in models.items():
        logger.info(f"Training {name}")
        start_time = time.time()
        try:
            kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            if name == "Gradient Boosting":
                param_distributions = {
                    'n_estimators': [50, 100],
                    'learning_rate': [0.1, 0.05],
                    'max_depth': [3, 5],
                    'subsample': [0.8, 0.9],
                    'max_features': ['sqrt', 'log2']
                }
                model = RandomizedSearchCV(model, param_distributions, n_iter=10, cv=kf, scoring='roc_auc', n_jobs=-1,
                                           verbose=1)
                model.fit(X_train_resampled, y_train_resampled)
                best_params = model.best_params_
                logger.info(f"Best params for Gradient Boosting: {best_params}")
                model = GradientBoostingClassifier(**best_params, random_state=42)
            cv_scores = cross_val_score(model, X_train_processed, y_train, cv=kf, scoring='roc_auc', n_jobs=-1)
            logger.info(f"Cross-validation scores for {name}: {cv_scores}")
            logger.info(f"Mean CV score for {name}: {cv_scores.mean():.2f}")

            model.fit(X_train_resampled, y_train_resampled)

            accuracy, precision, recall, f1, roc_auc, cm = evaluate_model(model, X_test_processed, y_test,name)

            end_time = time.time()
            elapsed_time = end_time - start_time

            logger.info(f'{name} Accuracy: {accuracy:.2f}')
            logger.info(f'{name} Precision: {precision:.2f}')
            logger.info(f'{name} Recall: {recall:.2f}')
            logger.info(f'{name} F1-Score: {f1:.2f}')
            logger.info(f'{name} ROC-AUC: {roc_auc:.2f}')
            logger.info(f'{name} Training time: {elapsed_time:.2f} seconds')
            logger.info('Confusion Matrix:')
            logger.info(cm)

            results.append({
                'Model': name,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1,
                'ROC-AUC': roc_auc,
                'Training Time (s)': elapsed_time
            })

            if hasattr(model, 'feature_importances_'):
                feature_importances[name] = model.feature_importances_

            # Save model
            dump(model, os.path.join(ROOT_DIR, f'models/{name.lower().replace(" ", "_")}_model.joblib'))
            logger.info(f"{name} model saved to models/{name.lower().replace(' ', '_')}_model.joblib")
        except Exception as e:
            logger.error(f"Error training {name}: {e}")

    # Voting Classifier
    logger.info("Training Voting Classifier")
    voting_clf = VotingClassifier(
        estimators=[
            ('rf', models["Random Forest"]),
            ('gb', models["Gradient Boosting"]),
            ('knn', models["KNN"]),
            ('xgb', models["XGBoost"]),
            ('lgbm', models["LightGBM"])
        ],
        voting='soft'
    )

    try:
        start_time = time.time()
        voting_clf.fit(X_train_resampled, y_train_resampled)
        accuracy, precision, recall, f1, roc_auc, cm = evaluate_model(voting_clf, X_test_processed, y_test,name)
        end_time = time.time()
        elapsed_time = end_time - start_time

        logger.info(f'Voting Classifier Accuracy: {accuracy:.2f}')
        logger.info(f'Voting Classifier Precision: {precision:.2f}')
        logger.info(f'Voting Classifier Recall: {recall:.2f}')
        logger.info(f'Voting Classifier F1-Score: {f1:.2f}')
        logger.info(f'Voting Classifier ROC-AUC: {roc_auc:.2f}')
        logger.info(f'Voting Classifier Training time: {elapsed_time:.2f} seconds')
        logger.info('Confusion Matrix:')
        logger.info(cm)

        results.append({
            'Model': 'Voting Classifier',
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'ROC-AUC': roc_auc,
            'Training Time (s)': elapsed_time
        })

        # Save model
        dump(voting_clf, os.path.join(ROOT_DIR, 'models/voting_classifier_model.joblib'))
        logger.info("Voting Classifier model saved to models/voting_classifier_model.joblib")
    except Exception as e:
        logger.error(f"Error training Voting Classifier: {e}")

    results_df = pd.DataFrame(results)
    results_df.set_index('Model', inplace=True)
    results_df.to_csv(os.path.join(ROOT_DIR, 'model_comparison.csv'))
    logger.info("Model Comparison:\n" + results_df.to_string())

    # Plot model comparison
    results_df[['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']].plot(kind='bar', figsize=(15, 8))
    plt.title('Model Comparison')
    plt.xlabel('Model')
    plt.ylabel('Score')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(ROOT_DIR, 'model_comparison.png'))
    plt.show()

    # Feature importances for top 20 features
    top_features = {}
    for name, importances in feature_importances.items():
        importances_df = pd.DataFrame({
            'Feature': X_train_processed.columns,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False).head(20)
        top_features[name] = importances_df
        # Define the directory path for saving the files
        feature_importance_dir = os.path.join(ROOT_DIR, 'feature_importance')
        os.makedirs(feature_importance_dir, exist_ok=True)
        # Save the feature importances CSV file
        importances_df.to_csv(
            os.path.join(feature_importance_dir, f'{name.lower().replace(" ", "_")}_feature_importances.csv'),
            index=False)

        plt.figure(figsize=(10, 6))
        plt.barh(importances_df['Feature'], importances_df['Importance'])
        plt.title(f'Feature Importances for {name}')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(os.path.join(feature_importance_dir, f'{name.lower().replace(" ", "_")}_feature_importances.png'))
        plt.show()

    # Print top features for each model
    for name, importances_df in top_features.items():
        logger.info(f"\nTop features for {name}:")
        logger.info(importances_df.to_string(index=False))


def main():
    try:
        features = pd.read_csv(os.path.join(ROOT_DIR, 'data/processed/url_features_reduced.csv'))
    except Exception as e:
        logger.error(f"Error reading the features file: {e}")
        return

    if features is None or features.empty:
        logger.error("Features data is empty or not loaded properly.")
        return

    logger.info("Features data loaded successfully.")
    logger.info(features.head())

    # sampled_features = features.sample(n=100000, random_state=42)
    #
    # try:
    #     X = sampled_features.drop(columns=['label'])  # Features
    #     y = sampled_features['label']  # Target variable
    # except KeyError as e:
    #     logger.error(f"Error in separating features and target: {e}")
    #     return

    try:
        X = features.drop(columns=['label'])  # Features
        y = features['label']  # Target variable
    except KeyError as e:
        logger.error(f"Error in separating features and target: {e}")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)

    logger.info(f"Number of training samples: {X_train.shape[0]}")
    logger.info(f"Number of testing samples: {X_test.shape[0]}")

    train_models(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()
