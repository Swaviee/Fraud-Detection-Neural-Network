# Fraud Detection with Neural Networks

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Machine learning imports
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve, \
    average_precision_score

# For handling class imbalance
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler

# For model explainability
import lime
import lime.lime_tabular

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Visualization settings
plt.style.use('ggplot')
sns.set(style="whitegrid", palette="muted", font_scale=1.2)
COLORS = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6"]


# --------------------------
# 1. Load and Preprocess Data
# --------------------------

def load_and_preprocess_data(file_path):
    """Load and preprocess the fraud dataset."""
    print(f"Loading data from {file_path}...")

    # Load the data
    fraud_df = pd.read_csv(file_path)

    print(f"Dataset shape: {fraud_df.shape}")
    print(f"Columns: {', '.join(fraud_df.columns)}")

    # Check class distribution
    if 'class' in fraud_df.columns:
        class_counts = fraud_df['class'].value_counts()
        print("\nClass distribution:")
        for cls, count in class_counts.items():
            percentage = count / len(fraud_df) * 100
            print(f"Class {cls}: {count} ({percentage:.2f}%)")

    # Handle missing/infinite values
    fraud_df = fraud_df.replace([np.inf, -np.inf], np.nan)

    # Check missing values
    missing_values = fraud_df.isnull().sum()
    missing_cols = missing_values[missing_values > 0]
    if not missing_cols.empty:
        print("\nColumns with missing values:")
        print(missing_cols)

    # Fill missing values
    # For numeric columns, fill with median
    numeric_cols = fraud_df.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        fraud_df[col] = fraud_df[col].fillna(fraud_df[col].median())

    # For categorical columns, fill with mode
    cat_cols = fraud_df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        if col != 'class':  # Skip target variable
            fraud_df[col] = fraud_df[col].fillna(fraud_df[col].mode()[0])

    # Basic data cleaning

    # Convert timestamp columns to datetime if they exist
    for col in ['signup_time', 'purchase_time']:
        if col in fraud_df.columns:
            fraud_df[col] = pd.to_datetime(fraud_df[col])

    # Feature engineering - create new features if timestamp columns exist
    if 'signup_time' in fraud_df.columns and 'purchase_time' in fraud_df.columns:
        # Time between signup and purchase
        fraud_df['purchase_time_since_signup'] = (
                    fraud_df['purchase_time'] - fraud_df['signup_time']).dt.total_seconds()

        # Extract hour of day
        fraud_df['signup_hour'] = fraud_df['signup_time'].dt.hour
        fraud_df['purchase_hour'] = fraud_df['purchase_time'].dt.hour

        # Same day purchase flag
        fraud_df['same_day_purchase'] = (fraud_df['purchase_time'].dt.date == fraud_df['signup_time'].dt.date).astype(
            int)

        # Day of week
        fraud_df['signup_dayofweek'] = fraud_df['signup_time'].dt.dayofweek
        fraud_df['purchase_dayofweek'] = fraud_df['purchase_time'].dt.dayofweek

        # Time of day categories
        def time_of_day(hour):
            if 5 <= hour < 12:
                return 'morning'
            elif 12 <= hour < 17:
                return 'afternoon'
            elif 17 <= hour < 22:
                return 'evening'
            else:
                return 'night'

        fraud_df['signup_timeofday'] = fraud_df['signup_hour'].apply(time_of_day)
        fraud_df['purchase_timeofday'] = fraud_df['purchase_hour'].apply(time_of_day)

    # Identify columns to drop (ID columns, original timestamps)
    columns_to_drop = []
    for col in ['user_id', 'signup_time', 'purchase_time', 'device_id', 'ip_address']:
        if col in fraud_df.columns:
            columns_to_drop.append(col)

    # Split features and target
    X = fraud_df.drop(columns_to_drop + ['class'], axis=1, errors='ignore')
    y = fraud_df['class']

    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    print(f"\nFeatures: {len(numerical_cols)} numerical, {len(categorical_cols)} categorical")

    # Create preprocessing pipeline - using sparse_output=False for compatibility with validation_split
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False),
             categorical_cols) if categorical_cols else ('pass', 'passthrough', [])
        ])

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Apply preprocessing
    print("Applying preprocessing...")
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    print(f"Processed training data shape: {X_train_processed.shape}")

    # Get feature names after one-hot encoding
    feature_names = numerical_cols.copy()
    if len(categorical_cols) > 0:
        for i, col in enumerate(categorical_cols):
            if hasattr(preprocessor.transformers_[1][1], 'categories_'):
                for category in preprocessor.transformers_[1][1].categories_[i]:
                    feature_names.append(f"{col}_{category}")

    # Keep original data for visualizations
    return X_train, X_test, X_train_processed, X_test_processed, y_train, y_test, feature_names, fraud_df, preprocessor


# --------------------------
# 2. Visualize Data
# --------------------------

def visualize_data(df):
    """Create visualizations of the fraud dataset."""
    print("Creating visualizations...")

    # Class distribution
    plt.figure(figsize=(8, 6))
    sns.countplot(x='class', data=df)
    plt.title('Class Distribution (0: Non-fraud, 1: Fraud)')
    plt.xticks([0, 1], ['Non-fraud', 'Fraud'])
    plt.show()

    # Age distribution by class
    if 'age' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(data=df, x='age', hue='class', bins=30, multiple='stack')
        plt.title('Age Distribution by Class')
        plt.show()

    # Purchase value distribution by class
    if 'purchase_value' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='class', y='purchase_value', data=df)
        plt.title('Purchase Value by Class')
        plt.xticks([0, 1], ['Non-fraud', 'Fraud'])
        plt.show()

    # Purchase time since signup by class
    if 'purchase_time_since_signup' in df.columns:
        plt.figure(figsize=(10, 6))
        df_filtered = df[
            df['purchase_time_since_signup'] < df['purchase_time_since_signup'].quantile(0.99)]  # Remove outliers
        sns.boxplot(x='class', y='purchase_time_since_signup', data=df_filtered)
        plt.title('Time Between Signup and Purchase by Class')
        plt.xticks([0, 1], ['Non-fraud', 'Fraud'])
        plt.ylabel('Time (seconds)')
        plt.show()

        # Also show log scale for better visualization
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='class', y='purchase_time_since_signup', data=df)
        plt.yscale('log')
        plt.title('Time Between Signup and Purchase by Class (log scale)')
        plt.xticks([0, 1], ['Non-fraud', 'Fraud'])
        plt.ylabel('Time (seconds, log scale)')
        plt.show()

    # Purchase hour distribution by class
    if 'purchase_hour' in df.columns:
        plt.figure(figsize=(12, 6))
        purchase_hour_by_class = pd.crosstab(df['purchase_hour'], df['class'], normalize='columns')
        purchase_hour_by_class.plot(kind='bar', stacked=False)
        plt.title('Purchase Hour Distribution by Class')
        plt.xlabel('Hour of Day')
        plt.ylabel('Proportion')
        plt.legend(['Non-fraud', 'Fraud'])
        plt.xticks(rotation=45)
        plt.show()

    # Browser or source distribution by class if they exist
    for cat_col in ['browser', 'source', 'sex']:
        if cat_col in df.columns:
            plt.figure(figsize=(12, 6))
            cat_by_class = pd.crosstab(df[cat_col], df['class'], normalize='columns')
            cat_by_class.plot(kind='bar', stacked=False)
            plt.title(f'{cat_col.capitalize()} Distribution by Class')
            plt.xlabel(cat_col.capitalize())
            plt.ylabel('Proportion')
            plt.legend(['Non-fraud', 'Fraud'])
            plt.xticks(rotation=45)
            plt.show()

    # Correlation heatmap for numerical columns
    num_cols = df.select_dtypes(include=['number']).columns
    if len(num_cols) > 2:  # Only create heatmap if we have enough numeric columns
        plt.figure(figsize=(12, 10))
        corr = df[num_cols].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))  # Create mask for upper triangle
        sns.heatmap(corr, annot=True, mask=mask, cmap='coolwarm',
                    vmin=-1, vmax=1, center=0, linewidths=0.5, fmt=".2f")
        plt.title('Correlation Matrix')
        plt.tight_layout()
        plt.show()

    # If we have time features, visualize signup/purchase patterns by day of week
    if all(col in df.columns for col in ['signup_dayofweek', 'purchase_dayofweek']):
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

        plt.figure(figsize=(12, 6))
        signup_day_by_class = pd.crosstab(df['signup_dayofweek'], df['class'], normalize='columns')
        signup_day_by_class.index = [days[i] for i in signup_day_by_class.index]
        signup_day_by_class.plot(kind='bar', stacked=False)
        plt.title('Signup Day of Week by Class')
        plt.xlabel('Day of Week')
        plt.ylabel('Proportion')
        plt.legend(['Non-fraud', 'Fraud'])
        plt.xticks(rotation=45)
        plt.show()

        plt.figure(figsize=(12, 6))
        purchase_day_by_class = pd.crosstab(df['purchase_dayofweek'], df['class'], normalize='columns')
        purchase_day_by_class.index = [days[i] for i in purchase_day_by_class.index]
        purchase_day_by_class.plot(kind='bar', stacked=False)
        plt.title('Purchase Day of Week by Class')
        plt.xlabel('Day of Week')
        plt.ylabel('Proportion')
        plt.legend(['Non-fraud', 'Fraud'])
        plt.xticks(rotation=45)
        plt.show()


# --------------------------
# 3. Model Building
# --------------------------

def build_model(input_dim):
    """Create a neural network model for fraud detection with regularization."""
    model = Sequential([
        # First hidden layer with L2 regularization
        Dense(64, activation='relu', kernel_regularizer=l2(0.001), input_dim=input_dim),
        Dropout(0.4),  # Increased dropout rate

        # Second hidden layer with L2 regularization
        Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.4),  # Increased dropout rate

        # Third hidden layer with L2 regularization
        Dense(16, activation='relu', kernel_regularizer=l2(0.001)),

        # Output layer
        Dense(1, activation='sigmoid')
    ])

    # Compile model
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )

    return model


# --------------------------
# 4. Training Function
# --------------------------

def train_model(model, X_train, y_train, X_val=None, y_val=None, epochs=50, batch_size=64, class_weight=None):
    """Train the model with validation data and early stopping."""
    # Define callbacks
    early_stopping = EarlyStopping(
        monitor='val_auc',
        patience=10,
        mode='max',
        restore_best_weights=True
    )

    # Set up validation data
    if X_val is None or y_val is None:
        # If no validation data provided, use validation_split
        X_train_internal, X_val_internal, y_train_internal, y_val_internal = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        validation_data = (X_val_internal, y_val_internal)
        training_data = X_train_internal
        training_labels = y_train_internal
    else:
        # Use provided validation data
        validation_data = (X_val, y_val)
        training_data = X_train
        training_labels = y_train

    # Verify validation data has both classes
    if len(np.unique(y_val if y_val is not None else y_val_internal)) < 2:
        print("WARNING: Validation data doesn't contain both classes!")

    # Train the model
    print(f"Training model with {len(training_labels)} samples...")
    print(f"Validation data has {np.unique(y_val if y_val is not None else y_val_internal, return_counts=True)}")

    history = model.fit(
        training_data, training_labels,
        validation_data=validation_data,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping],
        class_weight=class_weight,
        verbose=1
    )

    # Plot training history
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['auc'], label='Training AUC')
    if 'val_auc' in history.history:
        plt.plot(history.history['val_auc'], label='Validation AUC')
    plt.title('AUC Curves')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.legend()

    plt.tight_layout()
    plt.show()

    return history, model


# --------------------------
# 5. Evaluation Function
# --------------------------

def evaluate_model(model, X, y, name="Model"):
    """Evaluate model performance with multiple metrics."""
    # Make predictions
    print(f"Evaluating {name}...")
    y_pred_prob = model.predict(X, verbose=0)
    y_pred = (y_pred_prob > 0.5).astype(int)

    # Calculate metrics
    print(f"\n{name} Evaluation Metrics:")
    print(classification_report(y, y_pred))

    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
    plt.title(f'Confusion Matrix - {name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

    # ROC curve
    fpr, tpr, _ = roc_curve(y, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()

    # Precision-Recall curve (better for imbalanced data)
    precision, recall, _ = precision_recall_curve(y, y_pred_prob)
    avg_precision = average_precision_score(y, y_pred_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, lw=2, label=f'{name} (AP = {avg_precision:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="best")
    plt.show()

    # Return prediction probabilities for later use
    return y_pred_prob, roc_auc, avg_precision


# --------------------------
# 6. LIME Explainer Function
# --------------------------

def explain_with_lime(model, X_train_processed, X_test_processed, X_test, y_test, feature_names,
                      instances_to_explain=3):
    """Use LIME to explain model predictions."""
    print("\nGenerating LIME explanations...")

    # Create a LIME explainer
    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train_processed,
        mode="classification",
        feature_names=feature_names,
        class_names=["Non-Fraud", "Fraud"],
        discretize_continuous=True
    )

    # Select a few examples to explain (both correct and incorrect predictions)
    y_pred = (model.predict(X_test_processed, verbose=0) > 0.5).astype(int).flatten()
    correct_indices = np.where(y_pred == y_test.values)[0]
    incorrect_indices = np.where(y_pred != y_test.values)[0]

    # Get indices to explain
    indices_to_explain = []
    if len(correct_indices) > 0:
        indices_to_explain.extend(
            np.random.choice(correct_indices, min(instances_to_explain // 2, len(correct_indices)), replace=False))
    if len(incorrect_indices) > 0:
        indices_to_explain.extend(np.random.choice(incorrect_indices,
                                                   min(instances_to_explain - len(indices_to_explain),
                                                       len(incorrect_indices)), replace=False))

    # Generate explanations
    for idx in indices_to_explain:
        instance = X_test_processed[idx]
        prediction = model.predict(instance.reshape(1, -1), verbose=0)[0][0]
        true_label = y_test.iloc[idx]

        print(f"\nExplaining instance #{idx}")
        print(f"True label: {'Fraud' if true_label == 1 else 'Non-fraud'}")
        print(f"Predicted probability of fraud: {prediction:.4f}")
        print(f"Prediction: {'Fraud' if prediction > 0.5 else 'Non-fraud'}")

        # Generate explanation
        try:
            # Create a prediction function that returns probabilities for both classes
            def predict_fn(x):
                preds = model.predict(x, verbose=0)
                return np.hstack([1 - preds, preds])  # Return probabilities for [Non-fraud, Fraud]

            exp = explainer.explain_instance(
                instance,
                predict_fn,
                num_features=10
            )

            # Plot explanation
            plt.figure(figsize=(10, 6))
            exp.as_pyplot_figure()
            plt.title(
                f"LIME Explanation for Instance #{idx} (True: {'Fraud' if true_label == 1 else 'Non-fraud'}, Pred: {'Fraud' if prediction > 0.5 else 'Non-fraud'})")
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Error generating explanation: {e}")


# --------------------------
# 7. Main Function
# --------------------------

def main():
    # Load and preprocess data
    X_train, X_test, X_train_processed, X_test_processed, y_train, y_test, feature_names, fraud_df, preprocessor = load_and_preprocess_data(
        'Fraud_Data.csv')

    # Visualize data
    visualize_data(fraud_df)

    # Create validation splits
    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train_processed, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    print(f"Training set shape: {X_train_final.shape}")
    print(f"Validation set shape: {X_val.shape}")
    print(f"Training class distribution: {np.unique(y_train_final, return_counts=True)}")
    print(f"Validation class distribution: {np.unique(y_val, return_counts=True)}")

    # Get the input dimension for the model
    input_dim = X_train_processed.shape[1]

    # Build the baseline model
    print("\nBuilding baseline model...")
    baseline_model = build_model(input_dim)

    # Train the baseline model with explicit validation set
    baseline_history, baseline_model = train_model(baseline_model, X_train_final, y_train_final, X_val, y_val)

    # Evaluate the baseline model
    baseline_pred_prob, baseline_auc, baseline_ap = evaluate_model(baseline_model, X_test_processed, y_test,
                                                                   "Baseline Model")

    # Store results for comparison
    results = {
        'Model': ['Baseline'],
        'AUC': [baseline_auc],
        'AP': [baseline_ap]
    }

    # Implement random oversampling
    print("\nApplying random oversampling...")
    ros = RandomOverSampler(random_state=42)
    X_train_ros, y_train_ros = ros.fit_resample(X_train_final, y_train_final)
    print(f"After oversampling - X shape: {X_train_ros.shape}")
    print(f"Class distribution: {np.bincount(y_train_ros)}")

    # Build and train the oversampling model
    oversample_model = build_model(input_dim)
    oversample_history, oversample_model = train_model(oversample_model, X_train_ros, y_train_ros, X_val, y_val)

    # Evaluate the oversampling model
    oversample_pred_prob, oversample_auc, oversample_ap = evaluate_model(oversample_model, X_test_processed, y_test,
                                                                         "Oversampling Model")

    # Update results
    results['Model'].append('Oversampling')
    results['AUC'].append(oversample_auc)
    results['AP'].append(oversample_ap)

    # Implement random undersampling
    print("\nApplying random undersampling...")
    rus = RandomUnderSampler(random_state=42)
    X_train_rus, y_train_rus = rus.fit_resample(X_train_final, y_train_final)
    print(f"After undersampling - X shape: {X_train_rus.shape}")
    print(f"Class distribution: {np.bincount(y_train_rus)}")

    # Build and train the undersampling model
    undersample_model = build_model(input_dim)
    undersample_history, undersample_model = train_model(undersample_model, X_train_rus, y_train_rus, X_val, y_val)

    # Evaluate the undersampling model
    undersample_pred_prob, undersample_auc, undersample_ap = evaluate_model(undersample_model, X_test_processed, y_test,
                                                                            "Undersampling Model")

    # Update results
    results['Model'].append('Undersampling')
    results['AUC'].append(undersample_auc)
    results['AP'].append(undersample_ap)

    # Implement SMOTE
    print("\nApplying SMOTE...")
    smote = SMOTE(random_state=42)
    try:
        X_train_smote, y_train_smote = smote.fit_resample(X_train_final, y_train_final)
        print(f"After SMOTE - X shape: {X_train_smote.shape}")
        print(f"Class distribution: {np.bincount(y_train_smote)}")

        # Build and train the SMOTE model
        smote_model = build_model(input_dim)
        smote_history, smote_model = train_model(smote_model, X_train_smote, y_train_smote, X_val, y_val)

        # Evaluate the SMOTE model
        smote_pred_prob, smote_auc, smote_ap = evaluate_model(smote_model, X_test_processed, y_test, "SMOTE Model")

        # Update results
        results['Model'].append('SMOTE')
        results['AUC'].append(smote_auc)
        results['AP'].append(smote_ap)
    except Exception as e:
        print(f"Error applying SMOTE: {e}")
        print("Skipping SMOTE approach...")
        smote_pred_prob = None

    # Implement class weights
    print("\nApplying class weights...")
    # Calculate class weights based on the inverse of class frequencies
    neg_class_count = (y_train_final == 0).sum()
    pos_class_count = (y_train_final == 1).sum()
    total_samples = len(y_train_final)

    class_weights = {
        0: total_samples / (2.0 * neg_class_count),
        1: total_samples / (2.0 * pos_class_count)
    }
    print(f"Class weights: {class_weights}")

    # Build and train the class weight model
    class_weight_model = build_model(input_dim)
    class_weight_history, class_weight_model = train_model(class_weight_model, X_train_final, y_train_final, X_val,
                                                           y_val, class_weight=class_weights)

    # Evaluate the class weight model
    class_weight_pred_prob, class_weight_auc, class_weight_ap = evaluate_model(class_weight_model, X_test_processed,
                                                                               y_test, "Class Weight Model")

    # Update results
    results['Model'].append('Class Weights')
    results['AUC'].append(class_weight_auc)
    results['AP'].append(class_weight_ap)

    # Compare all models
    print("\nComparing all models...")

    # Compare ROC curves
    plt.figure(figsize=(10, 8))

    # Baseline
    fpr, tpr, _ = roc_curve(y_test, baseline_pred_prob)
    plt.plot(fpr, tpr, lw=2, label=f'Baseline (AUC = {baseline_auc:.3f})', color=COLORS[0])

    # Oversampling
    fpr, tpr, _ = roc_curve(y_test, oversample_pred_prob)
    plt.plot(fpr, tpr, lw=2, label=f'Oversampling (AUC = {oversample_auc:.3f})', color=COLORS[1])

    # Undersampling
    fpr, tpr, _ = roc_curve(y_test, undersample_pred_prob)
    plt.plot(fpr, tpr, lw=2, label=f'Undersampling (AUC = {undersample_auc:.3f})', color=COLORS[2])

    # SMOTE
    if 'smote_pred_prob' in locals() and smote_pred_prob is not None:
        fpr, tpr, _ = roc_curve(y_test, smote_pred_prob)
        plt.plot(fpr, tpr, lw=2, label=f'SMOTE (AUC = {smote_auc:.3f})', color=COLORS[3])

    # Class weights
    fpr, tpr, _ = roc_curve(y_test, class_weight_pred_prob)
    plt.plot(fpr, tpr, lw=2, label=f'Class Weights (AUC = {class_weight_auc:.3f})', color=COLORS[4])

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison')
    plt.legend(loc="lower right")
    plt.show()

    # Compare Precision-Recall curves
    plt.figure(figsize=(10, 8))

    # Baseline
    precision, recall, _ = precision_recall_curve(y_test, baseline_pred_prob)
    plt.plot(recall, precision, lw=2, label=f'Baseline (AP = {baseline_ap:.3f})', color=COLORS[0])

    # Oversampling
    precision, recall, _ = precision_recall_curve(y_test, oversample_pred_prob)
    plt.plot(recall, precision, lw=2, label=f'Oversampling (AP = {oversample_ap:.3f})', color=COLORS[1])

    # Undersampling
    precision, recall, _ = precision_recall_curve(y_test, undersample_pred_prob)
    plt.plot(recall, precision, lw=2, label=f'Undersampling (AP = {undersample_ap:.3f})', color=COLORS[2])

    # SMOTE
    if 'smote_pred_prob' in locals() and smote_pred_prob is not None:
        precision, recall, _ = precision_recall_curve(y_test, smote_pred_prob)
        plt.plot(recall, precision, lw=2, label=f'SMOTE (AP = {smote_ap:.3f})', color=COLORS[3])

    # Class weights
    precision, recall, _ = precision_recall_curve(y_test, class_weight_pred_prob)
    plt.plot(recall, precision, lw=2, label=f'Class Weights (AP = {class_weight_ap:.3f})', color=COLORS[4])

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve Comparison')
    plt.legend(loc="best")
    plt.show()

    # Print summary of results
    results_df = pd.DataFrame(results)
    print("\nModel Performance Summary:")
    print(results_df.sort_values('AUC', ascending=False))

    # Find best performing model
    best_model_idx = np.argmax(results['AUC'])
    best_model_name = results['Model'][best_model_idx]
    print(f"\nBest model based on AUC: {best_model_name}")

    # Use LIME to explain the best model
    if best_model_name == 'Baseline':
        best_model = baseline_model
    elif best_model_name == 'Oversampling':
        best_model = oversample_model
    elif best_model_name == 'Undersampling':
        best_model = undersample_model
    elif best_model_name == 'SMOTE' and 'smote_model' in locals():
        best_model = smote_model
    else:
        best_model = class_weight_model

    # Generate LIME explanations for the best model
    explain_with_lime(best_model, X_train_processed, X_test_processed, X_test, y_test, feature_names)

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()