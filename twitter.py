# Twitter Bot Detection with Neural Networks

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
    """Load and preprocess the Twitter bot dataset."""
    print(f"Loading data from {file_path}...")

    # Load the data
    twitter_df = pd.read_csv(file_path)

    print(f"Dataset shape: {twitter_df.shape}")
    if twitter_df.shape[1] > 10:
        print(f"First 10 columns: {', '.join(twitter_df.columns[:10])}...")
    else:
        print(f"Columns: {', '.join(twitter_df.columns)}")

    # Handle missing/infinite values
    twitter_df = twitter_df.replace([np.inf, -np.inf], np.nan)

    # Check missing values
    missing_values = twitter_df.isnull().sum()
    missing_cols = missing_values[missing_values > 0]
    if not missing_cols.empty:
        print("\nColumns with missing values:")
        print(missing_cols)

    # Fill missing values
    # For numeric columns, fill with median
    numeric_cols = twitter_df.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        twitter_df[col] = twitter_df[col].fillna(twitter_df[col].median())

    # For categorical columns, fill with mode
    cat_cols = twitter_df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        if col != 'account_type':  # Skip target variable
            twitter_df[col] = twitter_df[col].fillna(twitter_df[col].mode()[0])

    # Convert date columns if they exist
    if 'created_at' in twitter_df.columns:
        twitter_df['created_at'] = pd.to_datetime(twitter_df['created_at'], errors='coerce')

    # Create target variable (is_bot)
    if 'account_type' in twitter_df.columns:
        twitter_df['is_bot'] = (twitter_df['account_type'] == 'bot').astype(int)
        print("\nClass distribution:")
        print(twitter_df['is_bot'].value_counts())
        print(f"Bot percentage: {twitter_df['is_bot'].mean() * 100:.2f}%")

    # Identify columns to drop
    columns_to_drop = []
    for col in ['account_type', 'id', 'screen_name', 'profile_image_url',
                'profile_background_image_url', 'description']:
        if col in twitter_df.columns:
            columns_to_drop.append(col)

    # For demonstration, let's also drop columns with high missing values
    for col in twitter_df.columns:
        if twitter_df[col].isnull().mean() > 0.5:  # More than 50% missing
            if col not in columns_to_drop:
                columns_to_drop.append(col)

    # Split features and target
    X = twitter_df.drop(columns_to_drop + ['is_bot'], axis=1, errors='ignore')
    y = twitter_df['is_bot']

    # Handle location column separately if it exists (high cardinality)
    if 'location' in X.columns:
        # Just create a binary feature indicating if location is provided
        X['has_location'] = (~X['location'].isna() & (X['location'] != '')).astype(int)
        X = X.drop('location', axis=1)

    # Handle lang column separately if it exists (high cardinality)
    if 'lang' in X.columns and X['lang'].nunique() > 20:  # If too many languages
        # Only keep top 5 languages and group the rest
        top_langs = X['lang'].value_counts().nlargest(5).index
        X['lang'] = X['lang'].apply(lambda x: x if x in top_langs else 'other')

    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    print(f"\nFeatures: {len(numerical_cols)} numerical, {len(categorical_cols)} categorical")

    # Create preprocessing pipeline - FIX: Set sparse_output=False in OneHotEncoder
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
    return X_train, X_test, X_train_processed, X_test_processed, y_train, y_test, feature_names, twitter_df, preprocessor


# --------------------------
# 2. Visualize Data
# --------------------------

def visualize_data(df):
    """Create visualizations of the Twitter bot dataset."""
    print("Creating visualizations...")

    # Class distribution
    plt.figure(figsize=(8, 6))
    sns.countplot(x='is_bot', data=df)
    plt.title('Class Distribution (0: Human, 1: Bot)')
    plt.xticks([0, 1], ['Human', 'Bot'])
    plt.show()

    # Select key metrics for comparison
    key_metrics = ['followers_count', 'friends_count', 'statuses_count',
                   'favourites_count', 'average_tweets_per_day', 'account_age_days']

    # Keep only metrics that exist in the dataset
    key_metrics = [col for col in key_metrics if col in df.columns]

    # Boxplots for key metrics
    for metric in key_metrics:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='is_bot', y=metric, data=df)
        plt.title(f'{metric} by Account Type')
        plt.xticks([0, 1], ['Human', 'Bot'])
        plt.yscale('log')  # Log scale for better visualization
        plt.ylabel(f'{metric} (log scale)')
        plt.show()

    # Correlation matrix for numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 2:  # Only create heatmap if we have enough numeric columns
        plt.figure(figsize=(12, 10))
        corr = df[numeric_cols].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))  # Create mask for upper triangle
        sns.heatmap(corr, annot=False, mask=mask, cmap='coolwarm',
                    vmin=-1, vmax=1, center=0, linewidths=0.5)
        plt.title('Correlation Matrix')
        plt.tight_layout()
        plt.show()

    # Followers vs. Friends scatter plot
    if 'followers_count' in df.columns and 'friends_count' in df.columns:
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x='followers_count', y='friends_count',
                        hue='is_bot', data=df, alpha=0.5)
        plt.xscale('log')
        plt.yscale('log')
        plt.title('Followers vs. Friends by Account Type')
        plt.xlabel('Followers Count (log scale)')
        plt.ylabel('Friends Count (log scale)')
        plt.show()

    # Account creation pattern
    if 'created_at' in df.columns:
        df['creation_year'] = df['created_at'].dt.year
        plt.figure(figsize=(12, 6))
        sns.countplot(x='creation_year', hue='is_bot', data=df)
        plt.title('Account Creation by Year')
        plt.xlabel('Year')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.show()


# --------------------------
# 3. Model Building
# --------------------------

def build_model(input_dim):
    """Create a neural network model for bot detection with regularization."""
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
        class_names=["Human", "Bot"],
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
        print(f"True label: {'Bot' if true_label == 1 else 'Human'}")
        print(f"Predicted probability of being a bot: {prediction:.4f}")
        print(f"Prediction: {'Bot' if prediction > 0.5 else 'Human'}")

        # Generate explanation
        try:
            # Create a prediction function that returns probabilities for both classes
            def predict_fn(x):
                preds = model.predict(x, verbose=0)
                return np.hstack([1 - preds, preds])  # Return probabilities for [Human, Bot]

            exp = explainer.explain_instance(
                instance,
                predict_fn,
                num_features=10
            )

            # Plot explanation
            plt.figure(figsize=(10, 6))
            exp.as_pyplot_figure()
            plt.title(
                f"LIME Explanation for Instance #{idx} (True: {'Bot' if true_label == 1 else 'Human'}, Pred: {'Bot' if prediction > 0.5 else 'Human'})")
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Error generating explanation: {e}")


# --------------------------
# 7. Main Function
# --------------------------

def main():
    # Load and preprocess data
    X_train, X_test, X_train_processed, X_test_processed, y_train, y_test, feature_names, twitter_df, preprocessor = load_and_preprocess_data(
        'twitter_human_bots_dataset.csv')

    # Visualize data
    visualize_data(twitter_df)

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