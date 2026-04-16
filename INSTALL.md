# Instrukcja Instalacji - Adaptive Sentinel AI Factory

## Rozwiązywanie problemów z zależnościami

Jeśli napotkasz błędy podczas instalacji zależności, wykonaj następujące kroki:

### Krok 1: Użyj środowiska wirtualnego (ZALECANE)

```bash
# Utwórz środowisko wirtualne
python3 -m venv venv

# Aktywuj środowisko
source venv/bin/activate  # Na macOS/Linux
# lub
venv\Scripts\activate  # Na Windows
```

### Krok 2: Zaktualizuj pip, setuptools i wheel

```bash
pip install --upgrade pip setuptools wheel
```

### Krok 3: Instalacja zależności

**Opcja A: Tylko core (Django, kamera, kalibracja – bez treningu/detekcji)**  
Nie wymaga PyTorch; działa na Python 3.13 i innych środowiskach.

```bash
pip install -r requirements.txt
```

**Opcja B: Core + ML (trening YOLO, detekcja na żywo)**  
Po instalacji core zainstaluj PyTorch pod swoją platformę, potem ultralytics:

```bash
# Najpierw core
pip install -r requirements.txt

# PyTorch: wybierz wersję dla swojego systemu na https://pytorch.org/get-started/locally/
# Np. macOS / CPU:
pip install torch torchvision

# Potem ML
pip install -r requirements-ml.txt
```

**Opcja C: Instalacja krok po kroku (gdy są konflikty)**

```bash
pip install setuptools wheel
pip install Django channels daphne channels-redis
pip install numpy opencv-python
pip install Pillow psutil redis
# Opcjonalnie (trening/detekcja): pip install torch torchvision && pip install ultralytics
```

**Opcja C: Użyj pip z flagą --no-deps (jeśli są konflikty)**

```bash
pip install --no-deps Django channels daphne channels-redis
pip install numpy opencv-python ultralytics
pip install Pillow psutil redis
```

### Krok 4: Weryfikacja instalacji

```bash
python -c "import django; print('Django:', django.get_version())"
python -c "import cv2; print('OpenCV:', cv2.__version__)"
python -c "import numpy; print('NumPy:', numpy.__version__)"
python -c "import ultralytics; print('Ultralytics: OK')"
```

## Rozwiązywanie konkretnych problemów

### Problem: "Cannot import 'setuptools.build_meta'"

**Rozwiązanie:**
```bash
pip install --upgrade setuptools
# lub
pip install setuptools==68.0.0
```

### Problem: "numpy version conflict"

**Rozwiązanie:**
```bash
# Dla Python 3.13
pip install "numpy>=1.26.0"

# Dla Python 3.8-3.11 (Jetson)
pip install "numpy>=1.24.0,<1.26.0"
```

### Problem: "Django version conflict"

**Rozwiązanie:**
```bash
pip install "Django>=4.2,<5.0"
```

### Problem: "ultralytics dependencies"

**Rozwiązanie:**
```bash
# Ultralytics wymaga określonych wersji torch
pip install torch torchvision
pip install ultralytics
```

## Instalacja na NVIDIA Jetson

Na Jetson użyj `requirements-jetson.txt`:

```bash
pip install -r requirements-jetson.txt
```

Lub zainstaluj ręcznie z wersjami kompatybilnymi z Python 3.8-3.11:

```bash
pip install Django==4.2.7
pip install channels==4.0.0
pip install daphne==4.0.0
pip install numpy==1.24.3
pip install opencv-python==4.8.1.78
pip install ultralytics==8.0.196
pip install Pillow==10.1.0
pip install psutil==5.9.6
pip install redis==5.0.1
```

## Po instalacji

1. Uruchom migracje:
```bash
python manage.py makemigrations
python manage.py migrate
```

2. Utwórz superużytkownika (opcjonalnie):
```bash
python manage.py createsuperuser
```

3. Uruchom serwer:
```bash
python manage.py runserver
```

Lub z WebSockets:
```bash
daphne -b 0.0.0.0 -p 8000 sentinel_ai.asgi:application
```
