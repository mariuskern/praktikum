# Bewertung verschiedener Methoden hinsichtlich ihrer Fähigkeit abstrakte Merkmale aus 2D-Daten zu extrahieren

In diesem Projekt werden mehrere Methoden zur Extraktion von Merkmalen aus Bildern untersucht und miteinander verglichen. Im Mittelpunkt steht dabei die Qualität der extrahierten Merkmale, ihre Eignung für die Identifizierung ähnlicher Bilder und die Anwendbarkeit der Methoden auf verschiedene Datensätze.

## 🖼️ Datensätze
Im Repository sind Klassen und Hilfsmethoden für folgende Datensätze vorhanden:
- ImageNet: `dataset_utils/imagenet/`
- Places365: `dataset_utils/places365/`
- ArtPlaces: `dataset_utils/artplaces/`

## 🧠 Modelle
Das Repository enthält des weiteren Module für verschiedene Modelle:
- Perceptual Loss: `perceptual_loss/`
- Siamese Network: `siamese_network/`
- MoCo: `moco/`
- DINOv2: `dino_v2/`
- CLIP: `clip_model/`

## 🎯 Evaluationsmetriken
Für die Bewertung der Ergebnisse wurden folgende Metriken verwendet:
- Accuracy@k: Anteil der korrekt klassifizierten Bilder im Verhältnis zur Gesamtzahl der Bilder
- Precision@k: Anteil der korrekt klassifizierten Bilder innerhalb einer vorhergesagten Klasse
- Recall@k: Anteil der korrekt klassifizierten Bilder innerhalb der jeweiligen Klasse

## 🧪 Ablauf der Experimente
Für jede Methode und jeden Datensatz werden die folgenden Schritte durchgeführt:
1. Extraktion der Merkmale
2. Einfügen der Features in einen Faiss-Index
3. Suchen der n nächsten Nachbarn für jedes Bild eines Datensatzes
4. Bewerten der gefunden Bilder mithilfe verschiedener Metriken

## 📁 Projektstruktur (relevante Dateien/Ordner)
```
├─ clip_model/
├─ dataset_utils/               # Klassen zum Laden der Datensätze
│   ├─ artplaces/
│   ├─ imagenet/
│   └─ places365/
├─ dino_v2/
├─ distance_utils/
├─ moco/
├─ perceptual_loss/
├─ siamese_network/
├─ compare_vectors.ipynb
├─ confusion_matrix.ipynb
├─ evaluation_constants.py      # Konstanten für die Evaluation
├─ evaluation_utils.py          # Hilfsfunktionen für die Evaluation
└─ evaluation.ipynb             # Jupyter Notebook für die Evaluation
```

## ⚙️ Durchführung der Experimente

1. Abhängikeiten installieren
    
    Zunächst müssen alle Abhängigkeiten installiert werden. Zusätzlich ist die Installation von CLIP erforderlich. Weitere Informationen dazu finden sich in der README des Ordners `clip_model/`

2. Konfiguration prüfen

    In der Datei `evaluation_constants.py` müssen einige Einstellungen angepasst werden:
    - Speicherort für die Ergenisse
    - Speicherort der Modellgewichte

3. Datensatzpfade anpassen

    Des weiteren müssen die Pfade zu den entsprechenden Datensätzen in der Datei `evaluation.ipynb` durch die korrekten Pfade ersetzt werden

4. Experimente durchführen

    Die Datei `evaluation.ipynb` muss nun Zelle für Zelle ausgeführt werden.

5. Ergenisse einsehen

    Nach dem Ausführen der Experimente werden die Ergebnisse als JSON-Datei im zuvor definierten Ordner abgelegt.
