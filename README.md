# Adaptive Sentinel AI Factory - MLOps No-Code Platform

Kompleksowa aplikacja webowa do przeprowadzenia pełnego cyklu życia modelu AI do detekcji puszek na transporterze. Aplikacja działa na NVIDIA Jetson z wykorzystaniem CUDA.

## 🎯 Funkcjonalności

Aplikacja składa się z 5-etapowego wizarda:

1. **Konfiguracja Kamery**: Interfejs do podania adresu IP kamery Axis i podgląd na żywo przez OpenCV
2. **Kalibracja Tła**: Przycisk 'Pobierz wzorzec tła' - system zapisuje zdjęcia pustego pasu i wylicza bazowe parametry HSV
3. **Zbieranie Danych & Auto-labeling**: Automatyczne robienie zdjęć gdy wykryty zostanie ruch lub nowy kolor, z automatycznym generowaniem plików adnotacji YOLO TXT
4. **Trening Modelu (MLOps)**: Interfejs do ustawienia liczby epok i wyboru modelu bazowego (YOLOv8n/s/m/l/x), dashboard pokazujący postęp uczenia w czasie rzeczywistym
5. **Wdrożenie Live**: Przycisk 'URUCHOM PRODUKCJĘ' - system przełącza się na nowo wyuczony model i wyświetla dashboard detekcji z fioletowym podświetleniem i statusem 'CANS PRESENT'

## 🛠️ Wymagania techniczne

- Python 3.8+
- Django 4.2+
- Django Channels (WebSockets)
- Ultralytics YOLO
- OpenCV
- NVIDIA Jetson z CUDA support
- SQLite (baza danych)

## 📦 Instalacja

1. **Sklonuj repozytorium**:
```bash
cd /Users/radoslawtota/Development/LCM_AI
```

2. **Utwórz środowisko wirtualne**:
```bash
python3 -m venv venv
source venv/bin/activate  # Na Jetson: source venv/bin/activate
```

3. **Zainstaluj zależności**:
   
   **Na Mac/Linux z Python 3.13:**
   ```bash
   pip install -r requirements.txt
   ```
   
   **Na NVIDIA Jetson (Python 3.8-3.11):**
   ```bash
   pip install -r requirements-jetson.txt
   ```
   
   **Uwaga:** Jeśli wystąpi błąd z setuptools, najpierw zainstaluj:
   ```bash
   pip install --upgrade setuptools wheel
   ```

4. **Uruchom migracje**:
```bash
python manage.py makemigrations
python manage.py migrate
```

5. **Utwórz superużytkownika** (opcjonalnie):
```bash
python manage.py createsuperuser
```

6. **Uruchom serwer**:
```bash
python manage.py runserver 0.0.0.0:8000
```

Lub z użyciem Daphne (dla WebSockets):
```bash
daphne -b 0.0.0.0 -p 8000 sentinel_ai.asgi:application
```

## 📁 Struktura projektu

```
LCM_AI/
├── sentinel_ai/          # Główna konfiguracja Django
│   ├── settings.py       # Ustawienia projektu
│   ├── urls.py           # Główne URL-e
│   ├── asgi.py           # Konfiguracja ASGI dla WebSockets
│   └── routing.py        # Routing WebSocket
├── factory/              # Główna aplikacja
│   ├── models.py         # Modele Django (Camera, Dataset, ModelTraining, etc.)
│   ├── views.py          # Widoki API i wizard
│   ├── urls.py           # URL-e aplikacji
│   ├── consumers.py      # Konsumenci WebSocket
│   ├── camera_capture.py # Moduł przechwytywania kamery
│   ├── background_calibration.py  # Kalibracja tła i HSV
│   ├── auto_labeling.py  # Auto-labeling z logiką różnicową
│   └── training_manager.py # Zarządzanie treningiem YOLO
├── templates/            # Szablony HTML
│   └── factory/
│       └── wizard.html  # Główny interfejs wizarda
├── media/                # Pliki multimedialne
│   ├── datasets/        # Zbiory danych treningowych
│   ├── models/          # Wyuczone modele
│   └── backgrounds/     # Zdjęcia tła do kalibracji
└── requirements.txt     # Zależności Python
```

## 🚀 Użycie

### 1. Konfiguracja Kamery

1. Przejdź do kroku 1 wizarda
2. Wprowadź adres IP kamery Axis
3. Kliknij "Zapisz i Połącz"
4. Kliknij "Połącz z kamerą" aby rozpocząć podgląd na żywo

### 2. Kalibracja Tła

1. Przejdź do kroku 2
2. Upewnij się, że pas transporterowy jest pusty
3. Kliknij "Pobierz wzorzec tła"
4. System automatycznie obliczy parametry HSV

### 3. Zbieranie Danych

1. Przejdź do kroku 3
2. Utwórz nowy dataset
3. Kliknij "Start" aby rozpocząć automatyczne zbieranie danych
4. System będzie automatycznie:
   - Wykrywać ruch i nowe obiekty
   - Zbierać zdjęcia
   - Generować pliki adnotacji YOLO TXT

### 4. Trening Modelu

1. Przejdź do kroku 4
2. Wybierz dataset
3. Wybierz model bazowy (YOLOv8n - najszybszy, YOLOv8x - najdokładniejszy)
4. Ustaw liczbę epok i batch size
5. Kliknij "START TRAINING"
6. Obserwuj postęp na dashboardzie w czasie rzeczywistym

### 5. Wdrożenie Live

1. Przejdź do kroku 5
2. Wybierz wyuczony model (lub użyj best.pt)
3. Kliknij "URUCHOM PRODUKCJĘ"
4. Obserwuj detekcję na żywo z fioletowym podświetleniem obiektów

## 🔧 Konfiguracja dla NVIDIA Jetson

### Pamięć

Jetson ma współdzieloną pamięć (Unified Memory). Przed treningiem:
- Zamknij inne aplikacje
- Sprawdź dostępną pamięć w interfejsie
- System automatycznie ostrzeże o niskiej pamięci

### Model wyjściowy

Domyślnie aplikacja używa `best.pt` jako modelu wyjściowego. Po treningu nowy model będzie dostępny w:
- `media/models/training_{id}/run/weights/best.pt`

### Folder na dane

Zdjęcia do treningu są zapisywane w:
- `/media/datasets/{dataset_name}/images/train/`
- Adnotacje w: `/media/datasets/{dataset_name}/labels/train/`

## 📊 API Endpoints

### Kamera
- `POST /api/camera/create/` - Utwórz konfigurację kamery
- `POST /api/camera/<id>/connect/` - Połącz z kamerą
- `POST /api/camera/<id>/disconnect/` - Rozłącz kamerę
- `GET /api/camera/<id>/frame/` - Pobierz pojedynczą klatkę

### Kalibracja
- `POST /api/calibration/<camera_id>/capture/` - Przechwyć tło
- `POST /api/calibration/<camera_id>/calculate/` - Oblicz HSV

### Dataset
- `POST /api/dataset/create/` - Utwórz dataset
- `POST /api/dataset/<id>/start_collection/` - Rozpocznij zbieranie danych
- `POST /api/dataset/<id>/stop_collection/` - Zatrzymaj zbieranie

### Trening
- `POST /api/training/create/` - Utwórz konfigurację treningu
- `POST /api/training/<id>/start/` - Rozpocznij trening
- `GET /api/training/<id>/status/` - Status treningu

### Detekcja
- `POST /api/detection/start/` - Rozpocznij detekcję live
- `POST /api/detection/stop/` - Zatrzymaj detekcję
- `GET /api/detection/status/` - Status detekcji

## 🔌 WebSocket Endpoints

- `ws://host/ws/camera/<camera_id>/` - Stream kamery
- `ws://host/ws/training/<training_id>/` - Aktualizacje treningu
- `ws://host/ws/detection/` - Detekcja live

## 🎨 Interfejs

Interfejs wykorzystuje:
- **Tailwind CSS** - nowoczesny design
- **Ciemny motyw** - styl przemysłowy
- **Fioletowe akcenty** - podświetlenie obiektów
- **Duże przyciski akcji** - łatwa obsługa
- **Paski postępu** - wizualizacja procesów

## ⚠️ Uwagi

1. **Auto-labeling**: Wykorzystuje logikę różnicową (wszystko co nie jest tłem, jest puszką). Działa świetnie przy stabilnym tle.

2. **CUDA**: System automatycznie wykrywa dostępność CUDA. Jeśli CUDA nie jest dostępne, używa CPU (wolniejsze).

3. **Pamięć**: Przed treningiem sprawdź dostępną pamięć. System ostrzeże o niskiej pamięci.

4. **Model bazowy**: Domyślnie używany jest `best.pt`. Po treningu nowy model będzie dostępny w historii treningów.

## 📝 Licencja

Projekt stworzony dla Adaptive Sentinel AI Factory.

## 🤝 Wsparcie

W razie problemów sprawdź logi Django lub skontaktuj się z zespołem deweloperskim.
