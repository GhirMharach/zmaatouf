import os
import numpy as np
from skimage.feature import hog
from skimage import color
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

# Define the folders containing the images
train_folder = "C:\\Users\\Zayd Maatouf\\Documents\\2er Semster\\KI PROTOTYPE\\daten\\train2"
test_folder = "C:\\Users\\Zayd Maatouf\\Documents\\2er Semster\\KI PROTOTYPE\\daten\\test2"

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
    accuracies = []
    f1_scores = []
    recalls = []
    precisions = []
    
    for train_idx, val_idx in kfold.split(X_train, y_train):
        X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
        y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
        
        best_model.fit(X_fold_train, y_fold_train)
        y_pred = best_model.predict(X_fold_val)
        
        accuracies.append(accuracy_score(y_fold_val, y_pred))
        f1_scores.append(f1_score(y_fold_val, y_pred, average='macro', zero_division=0))
        recalls.append(recall_score(y_fold_val, y_pred, average='macro', zero_division=0))
        precisions.append(precision_score(y_fold_val, y_pred, average='macro', zero_division=0))
    
    avg_accuracy = np.mean(accuracies)
    avg_f1 = np.mean(f1_scores)
    avg_recall = np.mean(recalls)
    avg_precision = np.mean(precisions)
    
    print(f"RandomForestClassifier Cross-Validated Results:")
    print(f"Cross-Validated Accuracy: {avg_accuracy}")
    print(f"Cross-Validated F1-Score: {avg_f1}")
    print(f"Cross-Validated Recall: {avg_recall}")
    print(f"Cross-Validated Precision: {avg_precision}\n")
    
    # Output the best model
    print(f"\nThe best model is RandomForestClassifier with the following metrics:")
    print(f"Accuracy: {avg_accuracy}")
    print(f"F1-Score: {avg_f1}")
    print(f"Recall: {avg_recall}")
    print(f"Precision: {avg_precision}")

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


