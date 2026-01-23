import numpy as np
import csv

from batfloman_praktikum_lib.path_managment import validate_filename

def load_csv_oszi(filename):
    """
    Liest eine CSV-Datei vom Oszilloskop ein.
    
    Rückgabe:
        data     : list   # Liste der Messwerte (als float)
        metadata : dict   # Kopfbereich mit Infos (Sampling Period, Scale, etc.)
    """
    filename = validate_filename(filename, ".csv")

    metadata = {}
    data = []
    reading_data = False

    with open(filename, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:  # leere Zeilen überspringen
                continue

            # Beginn der Messdaten
            if "Waveform Data" in row[0]:
                reading_data = True
                continue

            if reading_data:
                # nur Zahlenzeilen
                try:
                    val = float(row[0])
                    data.append(val)
                except ValueError:
                    pass
            else:
                # Metadaten
                if len(row) >= 2:
                    key = row[0].strip()
                    value = row[1].strip()
                    metadata[key] = value

    return data, metadata

def load_csv_oszi_with_x(filename):
    """
    Liest Oszi-CSV und gibt (x, y, metadata) zurück.
    x = Zeitachse in Sekunden
    y = Messwerte (float)
    """
    filename = validate_filename(filename, ".csv")

    # load yData and metadata
    data, metadata = load_csv_oszi(filename)

    # calculate xData from sampling rate
    sampling_period = float(metadata.get("Sampling Period", 1.0))
    n = len(data)
    x = np.arange(n) * sampling_period

    return x, np.array(data), metadata
