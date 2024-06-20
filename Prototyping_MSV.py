# Import der erforderlichen Bibliotheken
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Laden des Digits-Datensatzes
digits = load_digits()

# Aufteilen der Daten in Features (X) und Labels (y)
X = digits.data
y = digits.target

# Aufteilen der Daten in Trainings- und Testsets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Skalierung der Daten
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialisierung des SVM-Modells
svm_model = SVC()

# Trainieren des Modells
svm_model.fit(X_train_scaled, y_train)

# Vorhersagen auf dem Testdatensatz
y_pred = svm_model.predict(X_test_scaled)

# Bewertung der Modellgenauigkeit
accuracy = accuracy_score(y_test, y_pred)
print("Genauigkeit des Modells:", accuracy)
