from PIL import Image
import os
import Augmentor


quali_10 = "10quali_train"
quali_50 = "50quali_train"
resize_quali_10 = "resize_10quali_train"
manipulate = "manipulierte Daten"
traindaten = "train1"
testdaten = "test1"
testmanipulate = "test_manipulate"

def komprimiern_kopie(input_folder, output_folder, compression_quality):

    # Liste aller Dateien im Eingabeordner
    file_list = os.listdir(input_folder)

    # Schleife über alle Dateien im Eingabeordner
    for file_name in file_list:
        # Bildpfad erstellen
        input_path = os.path.join(input_folder, file_name)
        
        # Bild öffnen
        img = Image.open(input_path)
            
        # Kopie des Bildes erstellen
        img_copy = img.copy()
            
        # Neuen Speicherort für das komprimierte Bild festlegen
        output_path = os.path.join(output_folder, file_name)
            
        # Bild mit Kompression speichern
        img_copy.save(output_path, quality=compression_quality)

    print("Komprimierung abgeschlossen.")


def resize_copy(neue_breite, neue_höhe,input_folder, output_folder):
    
    file_list = os.listdir(input_folder)

    for file_name in file_list:
        # Bildpfad erstellen
        input_path = os.path.join(input_folder, file_name)
        
        # Bild öffnen
        img = Image.open(input_path)
            
        # Kopie des Bildes erstellen
        img_copy = img.copy()
        resized_copy = img_copy.resize((neue_breite, neue_höhe))  
        # Neuen Speicherort für das komprimierte Bild festlegen
        output_path = os.path.join(output_folder, file_name)
            
        # Bild mit resize speichern
        resized_copy.save(output_path)

    print("rseized successfully")


#resize_copy(200, 100, quali_10, resize_quali_10)


def synthese(gen_daten_anzahl, input_folder, output_folder):

    #copy_original_images(input_folder, output_folder)

    # Erstelle ein Pipeline-Objekt mit dem Eingabeordner
    pipeline = Augmentor.Pipeline(input_folder, output_folder)

    # Füge Transformationen hinzu, z.B. Zufällige Rotationen und Spiegelungen
    pipeline.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)
    pipeline.flip_left_right(probability=0.5)
    
    # Helligkeit und Kotrast
    pipeline.random_brightness(probability=0.6, min_factor=0.7, max_factor=1.3)
    pipeline.random_contrast(probability=0.6, min_factor=0.7, max_factor=1.3)

    # Generiere die Bilder
    pipeline.sample(gen_daten_anzahl)

    # Bewege die generierten Bilder in den Ausgabeordner
    pipeline.process()
    
    print("synthesised successfully.")


synthese(482, resize_quali_10, "C:\\Users\\Miral Ibrahim\\OneDrive\\Desktop\\Uni\\Maschien_Learning\\Prototyping\\Daten\\kompromierte Daten\\neuetrain")



def umbenennung(ordner):
    true_count = 0
    false_count = 0

    for filename in os.listdir(ordner):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            if "True" in filename:
                true_count += 1
                neue_filename = f"{true_count}_True.png"
            elif "False" in filename:
                false_count += 1
                neue_filename = f"{false_count}_False.png"
            else:
                # Wenn weder "True" noch "False" im Dateinamen vorhanden ist, behalte den aktuellen Namen bei
                neue_filename = filename

            # Den neuen Dateinamen auf das Bild anwenden
            os.rename(os.path.join(ordner, filename), os.path.join(ordner, neue_filename))
    
    print("Umbenennung durchgeführt")


umbenennung("C:\\Users\\Miral Ibrahim\\OneDrive\\Desktop\\Uni\\Maschien_Learning\\Prototyping\\Daten\\kompromierte Daten\\neuetrain")

