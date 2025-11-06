import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

# Ordner, in dem die Bilddaten gespeichert sind
train_folder = "train1"
test_folder = "test1"


# Funktion zum Laden und Vorbereiten der Bilddaten
def load_images(folder, image_size):
    images = []
    labels = []
    
    for filename in os.listdir(folder):
                filepath = os.path.join(folder, filename)
                if os.path.isfile(filepath):
                    try:
                        img = Image.open(filepath)
                        img_array = np.array(img).flatten()
                        images.append(img_array)
                        label = filename.split('_')[1].split('.')[0]
                        labels.append(label)
                    except Exception as e:
                        print(f"Warnung: Bild {filename} konnte nicht geladen werden: {e}")
                else:
                    print(f"Warnung: Datei {filename} ist kein gültiges Bild.")
    return np.array(images), np.array(labels)


# Laden der Trainings- und Testdaten
X_train, y_train = load_images(train_folder, image_size=(640, 480))
X_test, y_test = load_images(test_folder, image_size=(640, 480))

# Überprüfen, ob die Daten nicht leer sind
if len(X_train) == 0 or len(X_test) == 0:
    print("Fehler: Keine Bilder gefunden. Stelle sicher, dass die Ordner nicht leer sind und dass die Bilddateien im richtigen Format vorliegen.")
else:
    # Modelle initialisieren
    models = [
        RandomForestClassifier(),
        GradientBoostingClassifier(),
        SVC(random_state=0.3,)
    ]
    
    best_model = None
    best_accuracy = 0
    best_f1_score = 0
    best_recall = 0
    best_precision = 0
    
    # Modelle trainieren und bewerten
    for model in models:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        precision = precision_score(y_test, y_pred, average='macro')
        
        print(f"Modell: {type(model).__name__}")
        print(f"Genauigkeit: {accuracy}")
        print(f"F1-Score: {f1}")
        print(f"Recall: {recall}")
        print(f"Precision: {precision}\n")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_f1_score = f1
            best_recall = recall
            best_precision = precision
    
    # Das beste Modell auswählen
    print(f"\nDas beste Modell ist {type(best_model).__name__} mit folgenden Metriken:")
    print(f"Genauigkeit: {best_accuracy}")
    print(f"F1-Score: {best_f1_score}")
    print(f"Recall: {best_recall}")
    print(f"Precision: {best_precision}")

