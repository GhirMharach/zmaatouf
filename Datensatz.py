import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

# Ordner, in dem die Bilddaten gespeichert sind
train_folder = 'C:\\Users\\ZAYD\\Documents\\1er Semestre KI\\Ki prototype\\train1\\manipulierte Daten'
test_folder = 'c:\\Users\\ZAYD\\Documents\\test1\\test1'

# Bildgröße
image_size = (640, 480)

# Funktion zum Laden und Vorbereiten der Bilddaten
def load_images(folder):
    images = []
    labels = []
    for class_name in os.listdir(folder):
        class_folder = os.path.join(folder, class_name)
        if os.path.isdir(class_folder):
            for filename in os.listdir(class_folder):
                # Check if the file exists and if it's an image
                filepath = os.path.join(class_folder, filename)
                if os.path.isfile(filepath) and (filename.lower().endswith('.jpg') or filename.lower().endswith('.png')):
                    img = cv2.imread(filepath)
                    if img is not None:
                        img = cv2.resize(img, image_size)
                        images.append(img.flatten())
                        labels.append(class_name)
                    else:
                        print(f"Warnung: Bild {filename} konnte nicht geladen werden.")
                else:
                    print(f"Warnung: Datei {filename} ist kein gültiges Bild.")
    return np.array(images), np.array(labels)


# Laden der Trainings- und Testdaten
X_train, y_train = load_images(train_folder)
X_test, y_test = load_images(test_folder)

# Überprüfen, ob die Daten nicht leer sind
if len(X_train) == 0 or len(X_test) == 0:
    print("Fehler: Keine Bilder gefunden. Stelle sicher, dass die Ordner nicht leer sind und dass die Bilddateien im richtigen Format vorliegen.")
else:
    # Modelle initialisieren
    models = [
        RandomForestClassifier(),
        GradientBoostingClassifier(),
        SVC()
    ]
    
    best_model = None
    best_accuracy = 0
    
    # Modelle trainieren und bewerten
    for model in models:
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
        print(f"Genauigkeit von {type(model).__name__}: {accuracy}")
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
    
    # Das beste Modell auswählen
    print(f"\nDas beste Modell ist {type(best_model).__name__} mit einer Genauigkeit von {best_accuracy}")