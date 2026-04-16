from django.db import models
from django.core.validators import MinValueValidator, MaxValueValidator
import os
from urllib.parse import urlparse, urlunparse
from django.conf import settings


class Camera(models.Model):
    """Camera configuration model"""
    name = models.CharField(max_length=200, default="Axis Camera")
    ip_address = models.GenericIPAddressField(unique=True)
    port = models.IntegerField(default=80)
    username = models.CharField(max_length=100, blank=True, null=True)
    password = models.CharField(max_length=100, blank=True, null=True)
    # Pełny adres strumienia (np. http://10.11.59.12/axis-cgi/mjpg/video.cgi?resolution=1920x1080)
    # Jeśli pusty, używany jest get_http_url() z resolution.
    stream_url = models.CharField(
        max_length=500, blank=True,
        help_text="Opcjonalnie: pełny URL MJPEG. Np. http://10.11.59.12/axis-cgi/mjpg/video.cgi?resolution=1920x1080"
    )
    resolution = models.CharField(
        max_length=20, default="", blank=True,
        help_text="Opcjonalnie: rozdzielczość w zapytaniu (?resolution=...). Puste = bez parametru."
    )
    is_active = models.BooleanField(default=False)
    # ROI: prostokąt (roi_x1..y2) LUB polygon (roi_polygon). Współrzędne 0–1.
    roi_x1 = models.FloatField(null=True, blank=True)
    roi_y1 = models.FloatField(null=True, blank=True)
    roi_x2 = models.FloatField(null=True, blank=True)
    roi_y2 = models.FloatField(null=True, blank=True)
    roi_polygon = models.JSONField(
        null=True, blank=True,
        help_text="Lista punktów [[x,y],[x,y],...] 0–1. Gdy ustawione, ma pierwszeństwo przed prostokątem."
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.name} ({self.ip_address})"

    def get_roi_normalized(self):
        """
        Zwraca None lub dict:
        - {'type': 'polygon', 'points': [[x,y],...]}
        - {'type': 'rect', 'rect': (x1,y1,x2,y2)}
        """
        if self.roi_polygon and len(self.roi_polygon) >= 3:
            return {'type': 'polygon', 'points': self.roi_polygon}
        if all(x is not None for x in (self.roi_x1, self.roi_y1, self.roi_x2, self.roi_y2)):
            return {'type': 'rect', 'rect': (self.roi_x1, self.roi_y1, self.roi_x2, self.roi_y2)}
        return None

    def get_roi_bbox(self):
        """Bounding box ROI (x1,y1,x2,y2) 0–1, do przycinania klatki. None jeśli brak ROI."""
        roi = self.get_roi_normalized()
        if roi is None:
            return None
        if roi['type'] == 'rect':
            return roi['rect']
        pts = roi['points']
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return (min(xs), min(ys), max(xs), max(ys))

    def get_rtsp_url(self):
        """Generate RTSP URL for Axis camera"""
        if self.username and self.password:
            return f"rtsp://{self.username}:{self.password}@{self.ip_address}:554/axis-media/media.amp"
        return f"rtsp://{self.ip_address}:554/axis-media/media.amp"

    def get_http_url(self):
        """URL strumienia MJPEG Axis. Źródło to stream_url LUB http://IP/axis-cgi/mjpg/video.cgi.
        Jeśli w adminie są username/password, wstawiane są do URL (np. http://user:pass@host/...)."""
        if self.stream_url and self.stream_url.strip():
            url = self.stream_url.strip()
        else:
            base = f"http://{self.ip_address}/axis-cgi/mjpg/video.cgi"
            res = (self.resolution or "").strip()
            url = f"{base}?resolution={res}" if res else base

        if self.username or self.password:
            try:
                parsed = urlparse(url)
                if not parsed.username and not parsed.password and parsed.hostname:
                    user = (self.username or "").replace(":", "%3A")
                    passw = (self.password or "").replace(":", "%3A")
                    netloc = f"{user}:{passw}@{parsed.hostname}" + (f":{parsed.port}" if parsed.port else "")
                    url = urlunparse((parsed.scheme, netloc, parsed.path, parsed.params, parsed.query, parsed.fragment))
            except Exception:
                pass  # zwróć url bez wstrzykiwania logowania
        return url


class BackgroundCalibration(models.Model):
    """Background calibration parameters. Wzorzec tła: plik, upload lub katalog ze zdjęciami."""
    camera = models.ForeignKey(Camera, on_delete=models.CASCADE, related_name='calibrations')
    calibration_date = models.DateTimeField(auto_now_add=True)
    # Ścieżka do pliku wzorca (katalog + plik na serwerze) lub ustawiana przy uploadzie
    background_image_path = models.CharField(max_length=500, blank=True)
    # Katalog ze zdjęciami tła – wszystkie obrazy (jpg/png) z tego katalogu są używane do HSV
    background_images_directory = models.CharField(
        max_length=500, blank=True,
        help_text="Ścieżka do katalogu na serwerze ze zdjęciami tła. Wszystkie jpg/png z katalogu będą użyte do kalibracji HSV."
    )
    # Opcjonalny upload w panelu admina – po zapisie ustawiamy background_image_path
    background_file = models.ImageField(upload_to='backgrounds/%Y/%m/', blank=True, null=True)
    hsv_min_h = models.IntegerField(default=0, validators=[MinValueValidator(0), MaxValueValidator(179)])
    hsv_min_s = models.IntegerField(default=0, validators=[MinValueValidator(0), MaxValueValidator(255)])
    hsv_min_v = models.IntegerField(default=0, validators=[MinValueValidator(0), MaxValueValidator(255)])
    hsv_max_h = models.IntegerField(default=179, validators=[MinValueValidator(0), MaxValueValidator(179)])
    hsv_max_s = models.IntegerField(default=255, validators=[MinValueValidator(0), MaxValueValidator(255)])
    hsv_max_v = models.IntegerField(default=255, validators=[MinValueValidator(0), MaxValueValidator(255)])
    is_active = models.BooleanField(default=True)

    def save(self, *args, **kwargs):
        super().save(*args, **kwargs)
        if self.background_file:
            path = getattr(self.background_file, 'path', None)
            if path:
                self.background_image_path = path
                super().save(update_fields=['background_image_path'])

    def __str__(self):
        return f"Calibration for {self.camera.name} - {self.calibration_date}"


class Dataset(models.Model):
    """Dataset for training"""
    name = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    dataset_path = models.CharField(max_length=500)
    images_count = models.IntegerField(default=0)
    labels_count = models.IntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.name} ({self.images_count} images)"


class DatasetImage(models.Model):
    """Individual image in dataset"""
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE, related_name='images')
    image_path = models.CharField(max_length=500)
    label_path = models.CharField(max_length=500, blank=True, null=True)
    has_label = models.BooleanField(default=False)
    captured_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return os.path.basename(self.image_path)


class ModelTraining(models.Model):
    """Model training configuration and results"""
    MODEL_CHOICES = [
        ('yolov8n', 'YOLOv8 Nano'),
        ('yolov8s', 'YOLOv8 Small'),
        ('yolov8m', 'YOLOv8 Medium'),
        ('yolov8l', 'YOLOv8 Large'),
        ('yolov8x', 'YOLOv8 XLarge'),
    ]

    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('running', 'Running'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
        ('cancelled', 'Cancelled'),
    ]

    name = models.CharField(max_length=200)
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE, related_name='trainings')
    base_model = models.CharField(max_length=20, choices=MODEL_CHOICES, default='yolov8n')
    epochs = models.IntegerField(default=100, validators=[MinValueValidator(1), MaxValueValidator(1000)])
    batch_size = models.IntegerField(default=16, validators=[MinValueValidator(1), MaxValueValidator(128)])
    image_size = models.IntegerField(default=640, validators=[MinValueValidator(320), MaxValueValidator(1280)])
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    model_path = models.CharField(max_length=500, blank=True, null=True)
    best_model_path = models.CharField(max_length=500, blank=True, null=True)
    training_log_path = models.CharField(max_length=500, blank=True, null=True)
    final_map50 = models.FloatField(blank=True, null=True)
    final_map50_95 = models.FloatField(blank=True, null=True)
    training_loss = models.FloatField(blank=True, null=True)
    validation_loss = models.FloatField(blank=True, null=True)
    started_at = models.DateTimeField(blank=True, null=True)
    completed_at = models.DateTimeField(blank=True, null=True)
    current_epoch = models.IntegerField(blank=True, null=True, help_text='Current epoch during training')
    progress_percent = models.IntegerField(blank=True, null=True, help_text='Training progress 0-100')
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.name} - {self.get_status_display()}"


class DetectionLog(models.Model):
    """Detection logs from live production"""
    camera = models.ForeignKey(Camera, on_delete=models.CASCADE, related_name='detections')
    model = models.ForeignKey(ModelTraining, on_delete=models.SET_NULL, null=True, blank=True, related_name='detections')
    image_path = models.CharField(max_length=500, blank=True, null=True)
    detection_count = models.IntegerField(default=0)
    confidence_avg = models.FloatField(blank=True, null=True)
    status = models.CharField(max_length=50, default='NO_CANS')
    detected_at = models.DateTimeField(auto_now_add=True)
    metadata = models.JSONField(default=dict, blank=True)

    class Meta:
        ordering = ['-detected_at']

    def __str__(self):
        return f"Detection {self.id} - {self.status} - {self.detected_at}"
