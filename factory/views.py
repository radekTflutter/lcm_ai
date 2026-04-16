"""
Views for Adaptive Sentinel AI Factory.
Heavy deps (cv2, numpy, channels, camera/calibration/training) are imported lazily
so that manage.py makemigrations/migrate work without opencv etc.
"""
import base64
import json
import logging
import threading
import time
from pathlib import Path

from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import render
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from .models import (
    Camera,
    BackgroundCalibration,
    Dataset,
    DatasetImage,
    ModelTraining,
    DetectionLog,
)

logger = logging.getLogger(__name__)

# Global state
_active_cameras = {}
_data_collection_active = {}
_training_managers = {}
_detection_active = False
_detection_thread = None
_detection_model = None


def wizard_view(request, stage=1):
    """Main wizard view. Bez importu torch/CUDA – status CUDA ładuje się w nav przez /api/system/status/."""
    try:
        stage = int(stage)
    except (ValueError, TypeError):
        stage = 1

    prev_stage = max(1, stage - 1)
    next_stage = min(5, stage + 1)
    progress_percent = stage * 20

    context = {
        'stage': stage,
        'prev_stage': prev_stage,
        'next_stage': next_stage,
        'progress_percent': progress_percent,
        'cameras': Camera.objects.all(),
        'datasets': Dataset.objects.all(),
        'trainings': ModelTraining.objects.all().order_by('-created_at')[:10],
    }
    return render(request, 'factory/wizard.html', context)


@require_http_methods(["GET"])
def get_system_status(request):
    """Return system/CUDA status for nav (base template)."""
    try:
        from .training_manager import check_cuda_available
    except ImportError:
        check_cuda_available = lambda: False
    return JsonResponse({'cuda_available': check_cuda_available()})


@csrf_exempt
@require_http_methods(["POST"])
def create_camera(request):
    """Create or update camera (stream_url = pełny URL MJPEG, np. http://IP/axis-cgi/mjpg/video.cgi)."""
    try:
        data = json.loads(request.body)
        ip = data['ip_address']
        stream_url = data.get('stream_url', '').strip()
        resolution = data.get('resolution', '').strip()
        defaults = {
            'name': data.get('name', 'Axis Camera'),
            'port': data.get('port', 80),
            'username': data.get('username', ''),
            'password': data.get('password', ''),
            'stream_url': stream_url,
            'resolution': resolution,
        }
        camera, created = Camera.objects.update_or_create(
            ip_address=ip,
            defaults=defaults,
        )
        return JsonResponse({'success': True, 'camera_id': camera.id})
    except Exception as e:
        logger.error(f"Error creating camera: {e}")
        return JsonResponse({'success': False, 'error': str(e)}, status=400)


@csrf_exempt
@require_http_methods(["POST"])
def connect_camera(request, camera_id):
    """Connect to camera and start streaming. Zawsze ładuje aktualną konfigurację z bazy (stream_url z admina)."""
    import traceback
    from .camera_capture import get_camera_instance, release_camera_instance

    try:
        camera = Camera.objects.get(id=camera_id)

        # Zawsze odśwież instancję (np. po zmianie stream_url w adminie)
        release_camera_instance(camera_id)
        camera_instance = get_camera_instance(camera_id)
        if not camera_instance:
            return JsonResponse({'success': False, 'error': 'Failed to create camera instance'}, status=400)

        # Connect
        if camera_instance.connect():
            camera_instance.start_capture()
            camera.is_active = True
            camera.save()
            _active_cameras[camera_id] = camera_instance
            return JsonResponse({'success': True})
        return JsonResponse({'success': False, 'error': 'Failed to connect to camera'}, status=400)

    except Camera.DoesNotExist:
        return JsonResponse({'success': False, 'error': 'Camera not found'}, status=404)
    except Exception as e:
        logger.exception("Error connecting camera: %s", e)
        tb = traceback.format_exc()
        return JsonResponse({
            'success': False,
            'error': str(e),
            'traceback': tb if settings.DEBUG else None,
        }, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def disconnect_camera(request, camera_id):
    """Disconnect camera"""
    from .camera_capture import release_camera_instance

    try:
        camera = Camera.objects.get(id=camera_id)
        camera.is_active = False
        camera.save()

        if camera_id in _active_cameras:
            _active_cameras[camera_id].stop_capture()
            del _active_cameras[camera_id]

        release_camera_instance(camera_id)
        return JsonResponse({'success': True})

    except Exception as e:
        logger.error(f"Error disconnecting camera: {e}")
        return JsonResponse({'success': False, 'error': str(e)}, status=400)


def get_camera_frame(request, camera_id):
    """Get single camera frame (GET or POST). Returns JSON with image or error."""
    import cv2
    from .camera_capture import get_camera_instance

    try:
        camera_instance = get_camera_instance(camera_id)
        if not camera_instance:
            return JsonResponse({
                'success': False,
                'error': 'Brak połączenia z kamerą. Sprawdź IP i Stream URL w panelu admina.'
            }, status=400)

        frame = camera_instance.get_frame()
        if frame is None:
            frame = camera_instance.capture_single()

        if frame is not None:
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            return JsonResponse({
                'success': True,
                'image': f'data:image/jpeg;base64,{frame_base64}'
            })
        return JsonResponse({
            'success': False,
            'error': 'Brak sygnału z kamery. Sprawdź zasilanie, sieć i adres MJPEG (Stream URL).'
        }, status=400)
    except Camera.DoesNotExist:
        return JsonResponse({'success': False, 'error': 'Kamera nie istnieje.'}, status=404)
    except Exception as e:
        logger.exception("Error getting frame for camera_id=%s: %s", camera_id, e)
        return JsonResponse({'success': False, 'error': str(e)}, status=400)


@require_http_methods(["GET"])
def get_camera_debug_url(request, camera_id):
    """Debug: zwraca efektywny URL kamery i ewentualny błąd. GET /api/camera/<id>/url/"""
    import traceback
    try:
        camera = Camera.objects.get(id=camera_id)
        url = camera.get_http_url()
        return JsonResponse({
            'camera_id': camera_id,
            'stream_url_from_db': (camera.stream_url or '').strip() or '(pusty)',
            'effective_url': url,
            'ip_address': camera.ip_address,
        })
    except Camera.DoesNotExist:
        return JsonResponse({'error': 'Camera not found'}, status=404)
    except Exception as e:
        logger.exception("get_camera_debug_url: %s", e)
        return JsonResponse({
            'error': str(e),
            'traceback': traceback.format_exc() if settings.DEBUG else None,
        }, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def set_camera_roi(request, camera_id):
    """Ustaw ROI: polygon (lista punktów) lub prostokąt. Body: {"polygon": [[x,y],...]} lub {"x1","y1","x2","y2"}."""
    try:
        camera = Camera.objects.get(id=camera_id)
        data = json.loads(request.body) if request.body else {}
        polygon = data.get('polygon')
        if polygon is not None and isinstance(polygon, list):
            if len(polygon) < 3:
                return JsonResponse({'success': False, 'error': 'Polygon musi mieć co najmniej 3 punkty.'}, status=400)
            points = []
            for p in polygon:
                if isinstance(p, (list, tuple)) and len(p) >= 2:
                    x, y = float(p[0]), float(p[1])
                    if not (0 <= x <= 1 and 0 <= y <= 1):
                        return JsonResponse({'success': False, 'error': 'Współrzędne punktów w zakresie 0–1.'}, status=400)
                    points.append([x, y])
            if len(points) < 3:
                return JsonResponse({'success': False, 'error': 'Polygon: co najmniej 3 punkty [x,y].'}, status=400)
            camera.roi_polygon = points
            camera.roi_x1 = camera.roi_y1 = camera.roi_x2 = camera.roi_y2 = None
            camera.save()
            return JsonResponse({'success': True, 'roi': {'type': 'polygon', 'points': points}})
        x1, y1, x2, y2 = data.get('x1'), data.get('y1'), data.get('x2'), data.get('y2')
        if x1 is not None and y1 is not None and x2 is not None and y2 is not None:
            x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
            if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
                return JsonResponse({'success': False, 'error': 'ROI: 0 ≤ x1 < x2 ≤ 1, 0 ≤ y1 < y2 ≤ 1'}, status=400)
            camera.roi_x1, camera.roi_y1, camera.roi_x2, camera.roi_y2 = x1, y1, x2, y2
            camera.roi_polygon = None
            camera.save()
            return JsonResponse({'success': True, 'roi': {'type': 'rect', 'rect': [x1, y1, x2, y2]}})
        return JsonResponse({'success': False, 'error': 'Podaj polygon: [[x,y],...] lub x1, y1, x2, y2 (0–1).'}, status=400)
    except Camera.DoesNotExist:
        return JsonResponse({'success': False, 'error': 'Camera not found'}, status=404)
    except Exception as e:
        logger.exception("set_camera_roi: %s", e)
        return JsonResponse({'success': False, 'error': str(e)}, status=400)


@require_http_methods(["GET"])
def get_camera_roi_status(request, camera_id):
    """Czy kamera ma ustawione ROI (polygon lub prostokąt). GET /api/camera/<id>/roi_status/"""
    try:
        camera = Camera.objects.get(id=camera_id)
        has_roi = False
        if getattr(camera, 'roi_polygon', None) and len(camera.roi_polygon) >= 3:
            has_roi = True
        elif all(getattr(camera, f, None) is not None for f in ('roi_x1', 'roi_y1', 'roi_x2', 'roi_y2')):
            has_roi = True
        return JsonResponse({'has_roi': has_roi})
    except Camera.DoesNotExist:
        return JsonResponse({'has_roi': False}, status=404)


@csrf_exempt
@require_http_methods(["POST"])
def capture_background(request, camera_id):
    """Capture background images for calibration. Preferuje get_frame() (stream), fallback capture_single()."""
    from .camera_capture import get_camera_instance
    from .background_calibration import calculate_hsv_parameters, save_background_images

    try:
        camera = Camera.objects.get(id=camera_id)
        camera_instance = get_camera_instance(camera_id)
        if not camera_instance:
            return JsonResponse({'success': False, 'error': 'Camera not connected'}, status=400)

        # Upewnij się, że stream działa (np. requests+Digest)
        if not getattr(camera_instance, 'is_running', False):
            if not camera_instance.connect():
                return JsonResponse({
                    'success': False,
                    'error': 'Nie udało się połączyć z kamerą. Użyj „Użyj wzorca z panelu admina” i wskaż katalog ze zdjęciami tła.',
                }, status=400)
            camera_instance.start_capture()
            for _ in range(60):
                if camera_instance.get_frame() is not None:
                    break
                time.sleep(0.15)

        images = []
        # Najpierw get_frame() (ze streamu requests/OpenCV) – stabilniejsze przy MJPEG
        for _ in range(15):
            frame = camera_instance.get_frame()
            if frame is not None:
                images.append(frame)
                if len(images) >= 10:
                    break
            time.sleep(0.2)
        # Jeśli mało klatek, spróbuj capture_single() (może działać przy niestandardowym MJPEG)
        while len(images) < 5:
            frame = camera_instance.capture_single()
            if frame is not None:
                images.append(frame)
            if len(images) >= 5:
                break
            time.sleep(0.2)

        if not images:
            return JsonResponse({
                'success': False,
                'error': 'Brak poprawnych klatek z kamery (błąd strumienia MJPEG). Użyj „Użyj wzorca z panelu admina” i wskaż katalog ze zdjęciami tła lub pojedynczy plik.',
            }, status=400)

        # Save images
        saved_paths = save_background_images(images, camera_id)

        # Calculate HSV parameters
        min_hsv, max_hsv = calculate_hsv_parameters(images)

        # Create calibration record
        calibration = BackgroundCalibration.objects.create(
            camera=camera,
            background_image_path=saved_paths[0] if saved_paths else '',
            hsv_min_h=min_hsv[0],
            hsv_min_s=min_hsv[1],
            hsv_min_v=min_hsv[2],
            hsv_max_h=max_hsv[0],
            hsv_max_s=max_hsv[1],
            hsv_max_v=max_hsv[2],
        )

        return JsonResponse({
            'success': True,
            'calibration_id': calibration.id,
            'hsv_min': [int(x) for x in min_hsv],
            'hsv_max': [int(x) for x in max_hsv],
            'images_captured': len(images)
        })

    except Exception as e:
        logger.error(f"Error capturing background: {e}")
        return JsonResponse({'success': False, 'error': str(e)}, status=400)


@csrf_exempt
@require_http_methods(["POST"])
def use_admin_background(request, camera_id):
    """Użyj wzorca tła z panelu admina: katalog (wszystkie zdjęcia), pojedynczy plik lub upload. Oblicza HSV."""
    from .background_calibration import (
        load_background_image,
        load_background_images_from_directory,
        calculate_hsv_parameters,
    )

    try:
        camera = Camera.objects.filter(id=camera_id).first()
        if not camera:
            return JsonResponse({'success': False, 'error': 'Kamera nie istnieje.'}, status=404)
        calibration = (
            BackgroundCalibration.objects.filter(camera_id=camera_id)
            .order_by('-calibration_date')
        ).first()
        if not calibration:
            return JsonResponse({
                'success': False,
                'error': 'Dla tej kamery nie ma kalibracji w panelu admina. Dodaj kalibrację i ustaw katalog ze zdjęciami tła lub plik.',
            }, status=400)

        images = []
        # 1) Katalog ze zdjęciami tła – wszystkie obrazy używane do HSV
        if getattr(calibration, 'background_images_directory', None) and calibration.background_images_directory.strip():
            dir_path = calibration.background_images_directory.strip()
            images = load_background_images_from_directory(dir_path)
            if not images:
                return JsonResponse({
                    'success': False,
                    'error': f'W katalogu {dir_path} nie znaleziono zdjęć (jpg/png).',
                }, status=400)
        # 2) Pojedynczy plik (ścieżka lub upload)
        if not images:
            image_path = None
            if getattr(calibration, 'background_file', None) and calibration.background_file:
                image_path = getattr(calibration.background_file, 'path', None)
            if not image_path and calibration.background_image_path:
                image_path = calibration.background_image_path
            if not image_path:
                return JsonResponse({
                    'success': False,
                    'error': 'Brak wzorca: ustaw w adminie "Katalog zdjęć tła" lub "Plik wzorca" (ścieżka/upload).',
                }, status=400)
            background = load_background_image(image_path)
            if background is None:
                return JsonResponse({'success': False, 'error': 'Nie można wczytać obrazu: ' + str(image_path)}, status=400)
            images = [background]
            calibration.background_image_path = image_path

        min_hsv, max_hsv = calculate_hsv_parameters(images)
        if not images and getattr(calibration, 'background_images_directory', None):
            calibration.background_image_path = calibration.background_images_directory.strip()
        calibration.hsv_min_h, calibration.hsv_min_s, calibration.hsv_min_v = min_hsv[0], min_hsv[1], min_hsv[2]
        calibration.hsv_max_h, calibration.hsv_max_s, calibration.hsv_max_v = max_hsv[0], max_hsv[1], max_hsv[2]
        calibration.is_active = True
        calibration.save()
        return JsonResponse({
            'success': True,
            'calibration_id': calibration.id,
            'hsv_min': [int(x) for x in min_hsv],
            'hsv_max': [int(x) for x in max_hsv],
            'images_used': len(images),
        })
    except Exception as e:
        logger.error("use_admin_background: %s", e)
        return JsonResponse({'success': False, 'error': str(e)}, status=400)


@csrf_exempt
@require_http_methods(["POST"])
def calculate_hsv(request, camera_id):
    """Recalculate HSV from existing background (directory or single image)."""
    from .background_calibration import (
        load_background_image,
        load_background_images_from_directory,
        calculate_hsv_parameters,
    )

    try:
        calibration = BackgroundCalibration.objects.filter(
            camera_id=camera_id,
            is_active=True
        ).first()

        if not calibration:
            return JsonResponse({'success': False, 'error': 'No active calibration found'}, status=404)

        images = []
        if getattr(calibration, 'background_images_directory', None) and calibration.background_images_directory.strip():
            images = load_background_images_from_directory(calibration.background_images_directory.strip())
        if not images:
            background = load_background_image(calibration.background_image_path)
            if background is None:
                return JsonResponse({'success': False, 'error': 'Failed to load background'}, status=400)
            images = [background]

        min_hsv, max_hsv = calculate_hsv_parameters(images)

        # Update calibration
        calibration.hsv_min_h = min_hsv[0]
        calibration.hsv_min_s = min_hsv[1]
        calibration.hsv_min_v = min_hsv[2]
        calibration.hsv_max_h = max_hsv[0]
        calibration.hsv_max_s = max_hsv[1]
        calibration.hsv_max_v = max_hsv[2]
        calibration.save()

        return JsonResponse({
            'success': True,
            'hsv_min': [int(x) for x in min_hsv],
            'hsv_max': [int(x) for x in max_hsv]
        })

    except Exception as e:
        logger.error(f"Error calculating HSV: {e}")
        return JsonResponse({'success': False, 'error': str(e)}, status=400)


@csrf_exempt
@require_http_methods(["POST"])
def create_dataset(request):
    """Create new dataset"""
    try:
        data = json.loads(request.body)
        dataset_name = data.get('name', f'Dataset_{timezone.now().strftime("%Y%m%d_%H%M%S")}')

        # Create dataset directory
        dataset_dir = Path(settings.DATASETS_ROOT) / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)

        # Create YOLO structure
        (dataset_dir / 'images' / 'train').mkdir(parents=True, exist_ok=True)
        (dataset_dir / 'images' / 'val').mkdir(parents=True, exist_ok=True)
        (dataset_dir / 'labels' / 'train').mkdir(parents=True, exist_ok=True)
        (dataset_dir / 'labels' / 'val').mkdir(parents=True, exist_ok=True)

        dataset = Dataset.objects.create(
            name=dataset_name,
            description=data.get('description', ''),
            dataset_path=str(dataset_dir)
        )

        return JsonResponse({'success': True, 'dataset_id': dataset.id})

    except Exception as e:
        logger.error(f"Error creating dataset: {e}")
        return JsonResponse({'success': False, 'error': str(e)}, status=400)


@require_http_methods(["GET", "POST"])
def get_dataset_stats(request, dataset_id):
    """Return current images_count and labels_count for a dataset (for wizard Statystyki). GET lub POST."""
    try:
        dataset = Dataset.objects.get(id=dataset_id)
        images_count = DatasetImage.objects.filter(dataset=dataset).count()
        labels_count = DatasetImage.objects.filter(dataset=dataset, has_label=True).count()
        return JsonResponse({
            'success': True,
            'images_count': images_count,
            'labels_count': labels_count,
        })
    except Dataset.DoesNotExist:
        return JsonResponse({'success': False, 'error': 'Dataset not found'}, status=404)


@csrf_exempt
@require_http_methods(["POST"])
def start_data_collection(request, dataset_id):
    """Start automatic data collection with auto-labeling. W body: camera_id, opcjonalnie roi. Używa ROI kamery jeśli ustawione."""
    import cv2
    from .camera_capture import get_camera_instance
    from .auto_labeling import detect_objects_differential, save_yolo_annotation
    from .background_calibration import get_one_background_for_detection

    try:
        dataset = Dataset.objects.get(id=dataset_id)
        data = json.loads(request.body) if request.body else {}
        camera_id = data.get('camera_id')
        if camera_id:
            camera = Camera.objects.filter(id=camera_id).first()
        else:
            camera = Camera.objects.filter(is_active=True).first()
        if not camera:
            return JsonResponse({'success': False, 'error': 'Wybierz kamerę w wizardzie lub ustaw aktywną w panelu admina.'}, status=400)

        calibration = BackgroundCalibration.objects.filter(
            camera=camera,
            is_active=True
        ).first()

        if not calibration:
            return JsonResponse({'success': False, 'error': 'No active calibration found'}, status=400)

        # Start collection thread
        def collection_loop():
            camera_instance = get_camera_instance(camera.id)
            if not camera_instance:
                return

            dataset_dir = Path(dataset.dataset_path)
            images_dir = dataset_dir / 'images' / 'train'
            labels_dir = dataset_dir / 'labels' / 'train'

            hsv_min = (calibration.hsv_min_h, calibration.hsv_min_s, calibration.hsv_min_v)
            hsv_max = (calibration.hsv_max_h, calibration.hsv_max_s, calibration.hsv_max_v)
            background = get_one_background_for_detection(calibration)
            if background is None:
                logger.error("Failed to load background for auto-labeling (ustaw plik lub katalog tła w kalibracji)")
                return
            roi = camera.get_roi_normalized()

            frame_count = 0
            last_motion_time = 0

            while dataset_id in _data_collection_active:
                frame = camera_instance.get_frame()
                if frame is None:
                    time.sleep(0.1)
                    continue

                # Detect motion/objects (tylko w ROI jeśli ustawione)
                try:
                    boxes = detect_objects_differential(frame, background, hsv_min, hsv_max, min_area=500, roi=roi)
                except TypeError:
                    boxes = detect_objects_differential(frame, background, hsv_min, hsv_max, min_area=500)

                current_time = time.time()
                # Capture if objects detected or every 5 seconds
                if boxes or (current_time - last_motion_time) > 5:
                    timestamp = int(time.time() * 1000)
                    image_path = images_dir / f'image_{timestamp}.jpg'
                    label_path = labels_dir / f'image_{timestamp}.txt'

                    # Save image
                    cv2.imwrite(str(image_path), frame)

                    # Auto-label
                    if boxes:
                        save_yolo_annotation(boxes, str(label_path))
                        has_label = True
                    else:
                        # Create empty label file
                        label_path.open('w').close()
                        has_label = False

                    # Create database record
                    DatasetImage.objects.create(
                        dataset=dataset,
                        image_path=str(image_path),
                        label_path=str(label_path),
                        has_label=has_label
                    )

                    dataset.images_count = DatasetImage.objects.filter(dataset=dataset).count()
                    dataset.labels_count = DatasetImage.objects.filter(dataset=dataset, has_label=True).count()
                    dataset.save()

                    frame_count += 1
                    last_motion_time = current_time

                    logger.info(f"Collected image {frame_count}: {len(boxes)} objects detected")

                time.sleep(0.5)  # Check every 0.5 seconds

        _data_collection_active[dataset_id] = True
        thread = threading.Thread(target=collection_loop, daemon=True)
        thread.start()

        return JsonResponse({'success': True, 'message': 'Data collection started'})

    except Exception as e:
        logger.error(f"Error starting data collection: {e}")
        return JsonResponse({'success': False, 'error': str(e)}, status=400)


@csrf_exempt
@require_http_methods(["POST"])
def stop_data_collection(request, dataset_id):
    """Stop data collection"""
    if dataset_id in _data_collection_active:
        del _data_collection_active[dataset_id]
    return JsonResponse({'success': True})


@csrf_exempt
@require_http_methods(["POST"])
def add_labeled_image(request, dataset_id):
    """Zapisz jedną klatkę + ręcznie podane boksy (YOLO). Opcjonalnie body.image (base64/data URL) – wtedy używany zamiast klatki z kamery."""
    import cv2
    import numpy as np
    from .camera_capture import get_camera_instance
    from .auto_labeling import save_yolo_annotation

    try:
        dataset = Dataset.objects.get(id=dataset_id)
        data = json.loads(request.body) if request.body else {}
        camera_id = data.get('camera_id')
        boxes_data = data.get('boxes', [])
        image_b64 = data.get('image')  # opcjonalnie: data:image/jpeg;base64,... lub sam base64
        if not camera_id:
            return JsonResponse({'success': False, 'error': 'Podaj camera_id.'}, status=400)
        camera = Camera.objects.filter(id=camera_id).first()
        if not camera:
            return JsonResponse({'success': False, 'error': 'Kamera nie istnieje.'}, status=404)

        frame = None
        if image_b64:
            try:
                if isinstance(image_b64, str) and image_b64.startswith('data:'):
                    image_b64 = image_b64.split(',', 1)[-1]
                raw = base64.b64decode(image_b64)
                nparr = np.frombuffer(raw, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            except Exception as e:
                logger.exception("add_labeled_image decode image: %s", e)
                return JsonResponse({'success': False, 'error': 'Nieprawidłowy obraz base64.'}, status=400)
        if frame is None:
            camera_instance = get_camera_instance(camera.id)
            if not camera_instance:
                return JsonResponse({'success': False, 'error': 'Połącz najpierw z kamerą.'}, status=400)
            frame = camera_instance.get_frame() or camera_instance.capture_single()
        if frame is None:
            return JsonResponse({'success': False, 'error': 'Nie udało się pobrać klatki z kamery.'}, status=400)

        dataset_dir = Path(dataset.dataset_path)
        images_dir = dataset_dir / 'images' / 'train'
        labels_dir = dataset_dir / 'labels' / 'train'
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        timestamp = int(time.time() * 1000)
        image_path = images_dir / f'image_{timestamp}.jpg'
        label_path = labels_dir / f'image_{timestamp}.txt'
        cv2.imwrite(str(image_path), frame)

        boxes_tuples = []
        for b in boxes_data:
            xc = float(b.get('x_center', 0))
            yc = float(b.get('y_center', 0))
            w = float(b.get('width', 0))
            h = float(b.get('height', 0))
            if w > 0 and h > 0:
                boxes_tuples.append((xc, yc, w, h))
        save_yolo_annotation(boxes_tuples, str(label_path), class_id=0)

        DatasetImage.objects.create(
            dataset=dataset,
            image_path=str(image_path),
            label_path=str(label_path),
            has_label=len(boxes_tuples) > 0,
        )
        dataset.images_count = DatasetImage.objects.filter(dataset=dataset).count()
        dataset.labels_count = DatasetImage.objects.filter(dataset=dataset, has_label=True).count()
        dataset.save()

        return JsonResponse({
            'success': True,
            'image_path': str(image_path),
            'boxes_count': len(boxes_tuples),
            'images_count': dataset.images_count,
            'labels_count': dataset.labels_count,
        })
    except Dataset.DoesNotExist:
        return JsonResponse({'success': False, 'error': 'Dataset not found'}, status=404)
    except Exception as e:
        logger.exception("add_labeled_image: %s", e)
        return JsonResponse({'success': False, 'error': str(e)}, status=400)


@csrf_exempt
@require_http_methods(["POST"])
def create_training(request):
    """Create training configuration"""
    try:
        data = json.loads(request.body)
        dataset = Dataset.objects.get(id=data['dataset_id'])

        training = ModelTraining.objects.create(
            name=data.get('name', f'Training_{timezone.now().strftime("%Y%m%d_%H%M%S")}'),
            dataset=dataset,
            base_model=data.get('base_model', 'yolov8n'),
            epochs=data.get('epochs', 100),
            batch_size=data.get('batch_size', 16),
            image_size=data.get('image_size', 640),
        )

        return JsonResponse({'success': True, 'training_id': training.id})

    except Exception as e:
        logger.error(f"Error creating training: {e}")
        return JsonResponse({'success': False, 'error': str(e)}, status=400)


@csrf_exempt
@require_http_methods(["POST"])
def start_training(request, training_id):
    """Start model training"""
    from asgiref.sync import async_to_sync
    from channels.layers import get_channel_layer
    from .training_manager import TrainingManager

    try:
        training = ModelTraining.objects.get(id=training_id)

        if training.status == 'running':
            return JsonResponse({'success': False, 'error': 'Training already running'}, status=400)

        # Create training manager
        def training_callback(data):
            channel_layer = get_channel_layer()
            async_to_sync(channel_layer.group_send)(
                f'training_{training_id}',
                {
                    'type': 'training_update',
                    'data': data
                }
            )

        manager = TrainingManager(training_id, callback=training_callback)
        _training_managers[training_id] = manager

        # Start training
        base_model_file = f"{training.base_model}.pt"
        manager.start_training(
            dataset_path=training.dataset.dataset_path,
            base_model=base_model_file,
            epochs=training.epochs,
            batch_size=training.batch_size,
            image_size=training.image_size
        )

        return JsonResponse({'success': True, 'message': 'Training started'})

    except Exception as e:
        logger.error(f"Error starting training: {e}")
        return JsonResponse({'success': False, 'error': str(e)}, status=400)


def get_training_status(request, training_id):
    """Get training status"""
    try:
        training = ModelTraining.objects.get(id=training_id)
        return JsonResponse({
            'success': True,
            'status': training.status,
            'map50': training.final_map50,
            'map50_95': training.final_map50_95,
            'best_model_path': training.best_model_path,
            'epoch': getattr(training, 'current_epoch', None),
            'progress': getattr(training, 'progress_percent', None),
        })
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=400)


@csrf_exempt
@require_http_methods(["POST"])
def start_detection(request):
    """Start live detection with trained model"""
    import cv2
    from asgiref.sync import async_to_sync
    from channels.layers import get_channel_layer
    from ultralytics import YOLO

    from .camera_capture import get_camera_instance
    from .training_manager import check_cuda_available

    global _detection_active, _detection_thread, _detection_model

    try:
        data = json.loads(request.body)
        training_id = data.get('training_id')

        if training_id:
            training = ModelTraining.objects.get(id=training_id)
            if not training.best_model_path:
                return JsonResponse({'success': False, 'error': 'No trained model available'}, status=400)
            model_path = training.best_model_path
        else:
            # Jeśli nie wybrano treningu, spróbuj użyć ostatniego zakończonego z best_model_path
            latest = ModelTraining.objects.filter(best_model_path__isnull=False).order_by('-created_at').first()
            if not latest or not latest.best_model_path:
                return JsonResponse({
                    'success': False,
                    'error': 'Brak domyślnego modelu best.pt. Najpierw zakończ trening w kroku 4 lub wybierz model z listy.'
                }, status=400)
            training_id = latest.id
            model_path = latest.best_model_path

        # Upewnij się, że plik modelu istnieje na tym serwerze
        from pathlib import Path
        model_path_obj = Path(model_path)
        if not model_path_obj.is_file():
            return JsonResponse({
                'success': False,
                'error': f'Plik modelu nie istnieje na tym serwerze: {model_path_obj}. '
                         f'Jeśli trenowałeś na innym komputerze (np. Windows C:\\...), skopiuj tutaj best.pt '
                         f'albo uruchom trening lokalnie i wybierz ten model w kroku 5.'
            }, status=400)

        # Load YOLO model
        _detection_model = YOLO(str(model_path_obj))
        if check_cuda_available():
            _detection_model.to('cuda')

        camera = Camera.objects.filter(is_active=True).first()
        if not camera:
            return JsonResponse({'success': False, 'error': 'Brak aktywnej kamery. W kroku 1 wybierz kamerę i kliknij „Połącz”.'}, status=400)

        # Sprawdź połączenie z kamerą PRZED startem wątku (unikamy 200 + cicha porażka przy timeout)
        camera_instance = get_camera_instance(camera.id)
        if not camera_instance:
            return JsonResponse({
                'success': False,
                'error': 'Nie można utworzyć połączenia z kamerą. Sprawdź konfigurację w panelu admina.'
            }, status=400)
        if not getattr(camera_instance, 'is_running', False):
            if not camera_instance.connect():
                return JsonResponse({
                    'success': False,
                    'error': f'Kamera {camera.ip_address} niedostępna (timeout lub błąd sieci). '
                             'Sprawdź: czy kamera jest w tej samej sieci, czy IP jest poprawne, firewall/port 80.'
                }, status=400)
            camera_instance.start_capture()
            # Krótkie czekanie na pierwszą klatkę
            for _ in range(30):
                if camera_instance.get_frame() is not None:
                    break
                time.sleep(0.1)

        def detection_loop():
            global _detection_active, _detection_model
            camera_instance = get_camera_instance(camera.id)
            if not camera_instance:
                logger.error("Detection: no camera instance for camera id=%s", camera.id)
                return
            if not getattr(camera_instance, 'is_running', False):
                try:
                    camera_instance.start_capture()
                except Exception as e:
                    logger.error("Detection: failed to start camera capture: %s", e)
                    return
            channel_layer = get_channel_layer()
            roi = camera.get_roi_normalized()
            from .auto_labeling import _apply_roi

            while _detection_active:
                frame = camera_instance.get_frame()
                if frame is None:
                    time.sleep(0.1)
                    continue

                h_full, w_full = frame.shape[:2]
                if roi:
                    crop, x_off, y_off, _, _ = _apply_roi(frame, roi)
                    if crop is None or crop.size == 0:
                        time.sleep(0.1)
                        continue
                    h_c, w_c = crop.shape[:2]
                    results = _detection_model(crop, verbose=False)
                    detections = []
                    for result in results:
                        for box in result.boxes:
                            x1, y1, x2, y2 = box.xyxy[0]
                            detections.append({
                                'x1': float(x1) + x_off,
                                'y1': float(y1) + y_off,
                                'x2': float(x2) + x_off,
                                'y2': float(y2) + y_off,
                                'confidence': float(box.conf[0]),
                                'class': int(box.cls[0])
                            })
                    annotated_crop = results[0].plot()
                    annotated_frame = frame.copy()
                    annotated_frame[y_off:y_off + h_c, x_off:x_off + w_c] = annotated_crop
                else:
                    results = _detection_model(frame, verbose=False)
                    detections = []
                    for result in results:
                        for box in result.boxes:
                            detections.append({
                                'x1': float(box.xyxy[0][0]),
                                'y1': float(box.xyxy[0][1]),
                                'x2': float(box.xyxy[0][2]),
                                'y2': float(box.xyxy[0][3]),
                                'confidence': float(box.conf[0]),
                                'class': int(box.cls[0])
                            })
                    annotated_frame = results[0].plot()

                _, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                frame_base64 = base64.b64encode(buffer).decode('utf-8')

                # Send update
                status = 'CANS PRESENT' if detections else 'NO_CANS'
                async_to_sync(channel_layer.group_send)(
                    'detection',
                    {
                        'type': 'detection_update',
                        'data': {
                            'type': 'detection',
                            'image': f'data:image/jpeg;base64,{frame_base64}',
                            'detections': detections,
                            'count': len(detections),
                            'status': status,
                            'timestamp': timezone.now().isoformat()
                        }
                    }
                )

                # Log detection
                if detections:
                    _training = ModelTraining.objects.filter(id=training_id).first() if training_id else None
                    DetectionLog.objects.create(
                        camera=camera,
                        model=_training,
                        detection_count=len(detections),
                        confidence_avg=sum(d['confidence'] for d in detections) / len(detections),
                        status=status,
                        metadata={'detections': detections}
                    )

                time.sleep(0.1)  # ~10 FPS

        _detection_active = True
        _detection_thread = threading.Thread(target=detection_loop, daemon=True)
        _detection_thread.start()

        return JsonResponse({'success': True, 'message': 'Detection started'})

    except Exception as e:
        logger.error(f"Error starting detection: {e}")
        return JsonResponse({'success': False, 'error': str(e)}, status=400)


@csrf_exempt
@require_http_methods(["POST"])
def stop_detection(request):
    """Stop live detection"""
    global _detection_active
    _detection_active = False
    return JsonResponse({'success': True})


def get_detection_status(request):
    """Get detection status"""
    return JsonResponse({
        'success': True,
        'active': _detection_active
    })
