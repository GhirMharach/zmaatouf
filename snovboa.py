import os
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from skimage.feature import hog
from skimage import color

# Define the folders containing the images
train_folder = "neue3"
test_folder = "test3"

# Function to load and prepare the image data
def load_images(folder, image_size=(640, 480)):
    images = []
    labels = []
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')  # Add valid image extensions here
    
    for filename in os.listdir(folder):
        if filename.lower().endswith(valid_extensions):
            filepath = os.path.join(folder, filename)
            try:
                img = Image.open(filepath).resize(image_size)
                img_gray = color.rgb2gray(np.array(img))  # Convert image to grayscale
                hog_features = hog(img_gray, pixels_per_cell=(16, 16), cells_per_block=(2, 2), feature_vector=True)
                images.append(hog_features)
                label = filename.split('_')[1].split('.')[0]
                labels.append(label)
            except Exception as e:
                print(f"Warning: Could not load image {filename}: {e}")
        else:
            print(f"Warning: File {filename} is not a valid image.")
    
    return np.array(images), np.array(labels)

# Verify the existence of the folders
if not os.path.exists(train_folder):
    print(f"Error: Training folder {train_folder} does not exist.")
if not os.path.exists(test_folder):
    print(f"Error: Test folder {test_folder} does not exist.")

# If the folders exist, proceed to load the images
if os.path.exists(train_folder) and os.path.exists(test_folder):
    # Load the training and test data
    X_train, y_train = load_images(train_folder)
    X_test, y_test = load_images(test_folder)

    # Check if the data is not empty
    if len(X_train) == 0 or len(X_test) == 0:
        print("Error: No images found. Ensure that the folders are not empty and the image files are in the correct format.")
    else:
        # Print class distribution
        unique, counts = np.unique(y_train, return_counts=True)
        print(f"Class distribution in training data: {dict(zip(unique, counts))}")
        
        # Define the parameter grid for RandomForestClassifier
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10]
        }

        # Initialize RandomForestClassifier
        model = RandomForestClassifier()

        # Initialize cross-validation
        kfold = StratifiedKFold(n_splits=5)
        
        # Perform Grid Search with cross-validation
        print("Starting Grid Search for RandomForestClassifier")
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=kfold, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        best_params = grid_search.best_params_
        print(f"Best parameters for RandomForestClassifier: {best_params}")
        
        best_model = grid_search.best_estimator_
        
        # Evaluate the best model found by Grid Search with cross-validation
        cv_results = {
            'Fold': [],
            'Accuracy': [],
            'F1-Score': [],
            'Recall': [],
            'Precision': []
        }
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train, y_train), 1):
            X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
            y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
            
            best_model.fit(X_fold_train, y_fold_train)
            y_pred = best_model.predict(X_fold_val)
            
            cv_results['Fold'].append(fold)
            cv_results['Accuracy'].append(accuracy_score(y_fold_val, y_pred))
            cv_results['F1-Score'].append(f1_score(y_fold_val, y_pred, average='macro', zero_division=0))
            cv_results['Recall'].append(recall_score(y_fold_val, y_pred, average='macro', zero_division=0))
            cv_results['Precision'].append(precision_score(y_fold_val, y_pred, average='macro', zero_division=0))
        
        # Convert cv_results to a DataFrame
        cv_results_df = pd.DataFrame(cv_results)
        print("\nCross-Validation Results:")
        print(cv_results_df)
        
        # Output the average results
        avg_accuracy = cv_results_df['Accuracy'].mean()
        avg_f1 = cv_results_df['F1-Score'].mean()
        avg_recall = cv_results_df['Recall'].mean()
        avg_precision = cv_results_df['Precision'].mean()
        
        print(f"\nThe best model is RandomForestClassifier with the following cross-validated metrics:")
        print(f"Average Accuracy: {avg_accuracy}")
        print(f"Average F1-Score: {avg_f1}")
        print(f"Average Recall: {avg_recall}")
        print(f"Average Precision: {avg_precision}")

        # Evaluate the best model on the test data
        best_model.fit(X_train, y_train)
        y_test_pred = best_model.predict(X_test)
        
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_f1 = f1_score(y_test, y_test_pred, average='macro', zero_division=0)
        test_recall = recall_score(y_test, y_test_pred, average='macro', zero_division=0)
        test_precision = precision_score(y_test, y_test_pred, average='macro', zero_division=0)
        
        print(f"\nTest Set Evaluation of the Best Model (RandomForestClassifier):")
        print(f"Accuracy: {test_accuracy}")
        print(f"F1-Score: {test_f1}")
        print(f"Recall: {test_recall}")
        print(f"Precision: {test_precision}")

        # Compute confusion matrix
        cm = confusion_matrix(y_test, y_test_pred)
        sns.heatmap(cm, annot=True, fmt='d')
        plt.title('Konfusionsmatrix')
        plt.ylabel('Tatsächliche Klasse')
        plt.xlabel('Vorhergesagte Klasse')
        plt.show()

        # Calculate and print TP, TN, FP, FN
        TP = cm[1, 1]
        TN = cm[0, 0]
        FP = cm[0, 1]
        FN = cm[1, 0]

        print(f'True Positives: {TP}')
        print(f'True Negatives: {TN}')
        print(f'False Positives: {FP}')
        print(f'False Negatives: {FN}')

