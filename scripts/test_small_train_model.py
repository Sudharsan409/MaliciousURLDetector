import logging
import os
import time

import matplotlib.pyplot as plt
import pandas as pd
from category_encoders import TargetEncoder
from imblearn.over_sampling import SMOTE
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    return accuracy, precision, recall, f1, roc_auc, cm


def train_models(X_train, X_test, y_train, y_test):
    logger.info("Fitting and transforming the training data")

    # Drop the 'url' column
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

    logger.info("Applying SMOTE to the training data")
    start_time = time.time()
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed, y_train)
    logger.info(f"SMOTE applied in {time.time() - start_time:.2f} seconds")

    os.makedirs(os.path.join(ROOT_DIR, 'models'), exist_ok=True)

    models = {
        "Random Forest": RandomForestClassifier(random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42)
    }

    results = []
    feature_importances = {}

    for name, model in models.items():
        logger.info(f"Training {name}")
        start_time = time.time()
        try:
            model.fit(X_train_resampled, y_train_resampled)
            accuracy, precision, recall, f1, roc_auc, cm = evaluate_model(model, X_test_processed, y_test)
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

    results_df = pd.DataFrame(results)
    results_df.set_index('Model', inplace=True)
    results_df.to_csv(os.path.join(ROOT_DIR, 'model_comparison.csv'))
    logger.info("Model Comparison:\n" + results_df.to_string())

    results_df.plot(kind='bar', figsize=(15, 8))
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
        importances_df.to_csv(os.path.join(ROOT_DIR, f'{name.lower().replace(" ", "_")}_feature_importances.csv'),
                              index=False)

        plt.figure(figsize=(10, 6))
        plt.barh(importances_df['Feature'], importances_df['Importance'])
        plt.title(f'Feature Importances for {name}')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(os.path.join(ROOT_DIR, f'{name.lower().replace(" ", "_")}_feature_importances.png'))
        plt.show()

    # Print top features for each model
    for name, importances_df in top_features.items():
        logger.info(f"\nTop features for {name}:")
        logger.info(importances_df.to_string(index=False))


def main():
    try:
        features = pd.read_csv(os.path.join(ROOT_DIR, 'data/processed/url_features.csv'))
    except Exception as e:
        logger.error(f"Error reading the features file: {e}")
        return

    if features is None or features.empty:
        logger.error("Features data is empty or not loaded properly.")
        return

    logger.info("Features data loaded successfully.")
    logger.info(features.head())

    try:
        X = features.drop(columns=['label'])  # Features
        y = features['label']  # Target variable
    except KeyError as e:
        logger.error(f"Error in separating features and target: {e}")
        return

    # Sample a smaller subset for quick testing
    X_sample = X.sample(n=1000, random_state=42)
    y_sample = y[X_sample.index]

    X_train, X_test, y_train, y_test = train_test_split(X_sample, y_sample, test_size=0.4, random_state=42,
                                                        stratify=y_sample)

    logger.info(f"Number of training samples: {X_train.shape[0]}")
    logger.info(f"Number of testing samples: {X_test.shape[0]}")

    train_models(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()
