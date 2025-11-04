# Bewertung verschiedener Methoden hinsichtlich ihrer Fähigkeit abstrakte Merkmale aus 2D-Daten zu extrahieren

In diesem Projekt werden mehrere Methoden zur Extraktion von Merkmalen aus Bildern untersucht und
miteinander verglichen. Im Mittelpunkt steht dabei die Qualität der extrahierten Merkmale,
ihre Eignung für die Identifizierung ähnlicher Bilder und die Anwendbarkeit der Methoden
auf verschiedene Datensätze.

## Datensätze
Im Repository sind Klassen und Hilfsmethoden für folgende Datensätze vorhanden:
- ArtPlaces: `dataset_utils/artplaces/`
- ImageNet: `dataset_utils/imagenet/`
- Places365: `dataset_utils/places365/`

## Modelle
Das Repository enthält des weiteren Module für verschiedene Modelle:
- Perceptual Loss: `perceptual_loss/`
- Siamese Network: `siamese_network/`
- MoCo: `moco/`
- CLIP: `clip_model/`
- DINOv2: `dino_v2/`

## Evaluationsmetriken
Für die Bewertung der Ergebnisse wurden folgende Metriken verwendet:
- Accuracy@k: Anteil der korrekt klassifizierten Bilder im Verhältnis zur Gesamtzahl der Bilder
- Precision@k: Anteil der korrekt klassifizierten Bilder innerhalb einer vorhergesagten Klasse
- Recall@k: Anteil der korrekt klassifizierten Bilder innerhalb der jeweiligen Klasse

## Ablauf der Experimente
Der Ablauf der Experimente war wie folgt:
- Extraktion der Merkmale aus Bildern der drei Datensätze mithilfe der verschiedenen Methoden
- Einfügen der Features in einen Faiss-Index
- Suchen der n nächsten Nachbarn für jedes Bild eines Datensatzes
- Bewerten der gefunden Bilder mithilfe verschiedener Metriken

## Projektstruktur (relevante Dateien/Ordner)
- `clip_model/`, `dino_v2/`, `moco/`, `perceptual_loss/`, `siamese_network/`: Modelle
- `dataset_utils/`: Klassen zum Laden der Datensätze
- `evaluation_utils.py`: Hilfsfunktionen für die Evaluation
- `evaluation_constants.py`: Konstanten für die Evaluation
- `evaluation.ipynb`: Jupyter Notebook für die Evaluation
- `plot_utils.py`: Hiflsmehtoden für die Visualisierung

## Durchführung der Experimente
Um die Experimente durchführen zu können, müssen zunächst alle Abhängigkeiten installiert werden. Zusätzlich muss CLIP installiert werden. Weitere Informationen dazu finden sich in der README des Ordners `clip_model/`. Nachdem alle Abhängigkeiten installiert wurden, kann die `evaluation.ipynb` Datei Zelle für Zelle ausgeführt werden. Die Ergebnisse werden als JSON im Ordner abgelegt, der zuvor in der `evaluation_constants.ipynb` festgelegt wurde. Neben dem Speicherort für die Ergebnisse kommen auch die Speicherorte der Gewichte in der `evaluation_constants.ipynb` vor. Diese müssen selbstverständlich ebenfalls ausgetauscht werden. Außerdem müssen die Pfade zu den entsprechenden Datensätzen in der `evaluation.ipynb` ausgetauscht werden.
