# Struktura Projektu - Adaptive Sentinel AI Factory

## 📁 Przegląd struktury

```
LCM_AI/
├── sentinel_ai/              # Główna konfiguracja Django
│   ├── __init__.py
│   ├── settings.py           # Ustawienia projektu, baza danych, Channels
│   ├── urls.py               # Główne routing URL
│   ├── asgi.py               # Konfiguracja ASGI dla WebSockets
│   ├── wsgi.py               # Konfiguracja WSGI
│   └── routing.py            # Routing WebSocket (camera, training, detection)
│
├── factory/                  # Główna aplikacja
│   ├── __init__.py
│   ├── apps.py               # Konfiguracja aplikacji
│   ├── admin.py              # Panel administracyjny Django
│   ├── models.py             # Modele Django:
│   │                          #   - Camera
│   │                          #   - BackgroundCalibration
│   │                          #   - Dataset, DatasetImage
│   │                          #   - ModelTraining
│   │                          #   - DetectionLog
│   ├── views.py              # Widoki API i wizard (5 etapów)
│   ├── urls.py               # Routing aplikacji
│   ├── consumers.py          # Konsumenci WebSocket:
│   │                          #   - CameraConsumer
│   │                          #   - TrainingConsumer
│   │                          #   - DetectionConsumer
│   ├── camera_capture.py     # Moduł przechwytywania kamery Axis
│   ├── background_calibration.py  # Kalibracja tła i obliczanie HSV
│   ├── auto_labeling.py      # Auto-labeling z logiką różnicową
│   └── training_manager.py    # Zarządzanie treningiem YOLO z CUDA
│
├── templates/                # Szablony HTML
│   ├── base.html             # Bazowy szablon z Tailwind CSS
│   └── factory/
│       └── wizard.html       # Główny interfejs 5-etapowego wizarda
│
├── media/                    # Pliki multimedialne (tworzone automatycznie)
│   ├── datasets/            # Zbiory danych treningowych
│   │   └── {dataset_name}/
│   │       ├── images/
│   │       │   ├── train/
│   │       │   └── val/
│   │       └── labels/
│   │           ├── train/
│   │           └── val/
│   ├── models/              # Wyuczone modele
│   │   └── training_{id}/
│   │       └── run/
│   │           └── weights/
│   │               └── best.pt
│   └── backgrounds/         # Zdjęcia tła do kalibracji
│       └── camera_{id}/
│
├── manage.py                 # Django management script
├── requirements.txt          # Zależności Python
├── start.sh                  # Skrypt startowy
├── README.md                 # Dokumentacja główna
└── .gitignore               # Pliki ignorowane przez git
```

## 🔧 Główne komponenty

### 1. Modele Django (`factory/models.py`)

- **Camera**: Konfiguracja kamery Axis (IP, port, credentials)
- **BackgroundCalibration**: Parametry HSV z kalibracji tła
- **Dataset**: Zbiór danych treningowych
- **DatasetImage**: Pojedyncze zdjęcie z adnotacją
- **ModelTraining**: Konfiguracja i wyniki treningu
- **DetectionLog**: Logi detekcji z produkcji

### 2. Moduły funkcjonalne

#### `camera_capture.py`
- Klasa `CameraCapture`: Thread-safe przechwytywanie klatek
- Obsługa RTSP i HTTP MJPEG dla kamer Axis
- Globalne zarządzanie instancjami kamer

#### `background_calibration.py`
- `calculate_hsv_parameters()`: Obliczanie min/max HSV z obrazów tła
- `save_background_images()`: Zapis zdjęć tła
- `load_background_image()`: Ładowanie obrazu tła

#### `auto_labeling.py`
- `detect_objects_differential()`: Wykrywanie obiektów przez różnicę z tłem
- `save_yolo_annotation()`: Zapis adnotacji w formacie YOLO TXT
- `auto_label_image()`: Automatyczne etykietowanie pojedynczego obrazu

#### `training_manager.py`
- Klasa `TrainingManager`: Zarządzanie treningiem YOLO
- `check_cuda_available()`: Sprawdzanie dostępności CUDA
- `get_available_memory_mb()`: Sprawdzanie dostępnej pamięci
- Integracja z Ultralytics YOLO

### 3. Widoki API (`factory/views.py`)

#### Kamera
- `create_camera()`: Utworzenie konfiguracji
- `connect_camera()`: Połączenie z kamerą
- `disconnect_camera()`: Rozłączenie
- `get_camera_frame()`: Pobranie pojedynczej klatki

#### Kalibracja
- `capture_background()`: Przechwycenie tła
- `calculate_hsv()`: Obliczenie parametrów HSV

#### Dataset
- `create_dataset()`: Utworzenie datasetu
- `start_data_collection()`: Rozpoczęcie zbierania danych z auto-labeling
- `stop_data_collection()`: Zatrzymanie zbierania

#### Trening
- `create_training()`: Utworzenie konfiguracji treningu
- `start_training()`: Rozpoczęcie treningu
- `get_training_status()`: Status treningu

#### Detekcja
- `start_detection()`: Rozpoczęcie detekcji live
- `stop_detection()`: Zatrzymanie detekcji
- `get_detection_status()`: Status detekcji

### 4. WebSocket Consumers (`factory/consumers.py`)

- `CameraConsumer`: Stream kamery w czasie rzeczywistym
- `TrainingConsumer`: Aktualizacje postępu treningu
- `DetectionConsumer`: Detekcja live z podświetleniem

### 5. Frontend (`templates/`)

- **base.html**: Bazowy szablon z Tailwind CSS, ciemny motyw
- **wizard.html**: 5-etapowy wizard z:
  - Krok 1: Konfiguracja kamery z podglądem
  - Krok 2: Kalibracja tła z wyświetlaniem HSV
  - Krok 3: Zbieranie danych z auto-labeling
  - Krok 4: Trening z dashboardem w czasie rzeczywistym
  - Krok 5: Wdrożenie live z fioletowym podświetleniem

## 🚀 Przepływ pracy

1. **Konfiguracja kamery** → Połączenie z kamerą Axis
2. **Kalibracja tła** → Przechwycenie pustego pasa, obliczenie HSV
3. **Zbieranie danych** → Automatyczne zbieranie z auto-labeling
4. **Trening** → Trening YOLO z CUDA, monitoring w czasie rzeczywistym
5. **Wdrożenie** → Detekcja live z podświetleniem obiektów

## 🔌 Integracje

- **OpenCV**: Przechwytywanie i przetwarzanie obrazów
- **Ultralytics YOLO**: Trening i detekcja
- **Django Channels**: WebSockets dla czasu rzeczywistego
- **CUDA**: Akceleracja na NVIDIA Jetson
- **SQLite**: Baza danych

## 📝 Uwagi implementacyjne

1. **Pamięć Jetson**: System sprawdza dostępną pamięć przed treningiem
2. **Model wyjściowy**: Domyślnie `best.pt`, po treningu dostępny w historii
3. **Auto-labeling**: Logika różnicowa (wszystko co nie jest tłem = puszka)
4. **Threading**: Wszystkie długotrwałe operacje w tle
5. **WebSockets**: Real-time updates dla kamery, treningu i detekcji
