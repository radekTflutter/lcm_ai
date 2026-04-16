from django.contrib import admin
from .models import Camera, BackgroundCalibration, Dataset, DatasetImage, ModelTraining, DetectionLog


@admin.register(Camera)
class CameraAdmin(admin.ModelAdmin):
    list_display = ['name', 'ip_address', 'port', 'resolution', 'has_roi', 'is_active', 'created_at']
    list_filter = ['is_active', 'created_at']
    search_fields = ['name', 'ip_address']
    fieldsets = (
        (None, {
            'fields': ('name', 'ip_address', 'port', 'username', 'password'),
        }),
        ('Strumień MJPEG (Axis)', {
            'fields': ('stream_url', 'resolution'),
            'description': 'Pełny URL lub IP + resolution. Np. http://10.11.59.12/axis-cgi/mjpg/video.cgi?resolution=1920x1080',
        }),
        ('ROI (obszar wykrywania)', {
            'fields': ('roi_x1', 'roi_y1', 'roi_x2', 'roi_y2'),
            'description': 'Opcjonalnie: prostokąt 0–1. Puszki wykrywane tylko w tym obszarze (ustaw w wizardzie krok 3).',
        }),
        ('Status', {
            'fields': ('is_active', 'url_used'),
        }),
    )
    readonly_fields = ['url_used']

    def has_roi(self, obj):
        return all(x is not None for x in (obj.roi_x1, obj.roi_y1, obj.roi_x2, obj.roi_y2))
    has_roi.boolean = True
    has_roi.short_description = 'ROI'

    def url_used(self, obj):
        """Adres używany do połączenia (get_http_url())."""
        return obj.get_http_url() if obj.pk else '—'
    url_used.short_description = 'Adres strumienia (używany przy połączeniu)'


@admin.register(BackgroundCalibration)
class BackgroundCalibrationAdmin(admin.ModelAdmin):
    list_display = ['camera', 'calibration_date', 'is_active', 'background_image_path_short']
    list_filter = ['is_active', 'calibration_date']
    search_fields = ['camera__name']
    fieldsets = (
        (None, {'fields': ('camera', 'is_active')}),
        ('Wzorzec tła', {
            'fields': ('background_images_directory', 'background_file', 'background_image_path'),
            'description': 'Katalog ze zdjęciami tła: ścieżka do folderu na serwerze – wszystkie jpg/png będą użyte do HSV. Albo pojedynczy plik (upload lub ścieżka).',
        }),
        ('Parametry HSV (obliczane przy „Pobierz wzorzec” w aplikacji)', {
            'fields': (
                'hsv_min_h', 'hsv_min_s', 'hsv_min_v',
                'hsv_max_h', 'hsv_max_s', 'hsv_max_v',
            ),
        }),
    )

    def background_image_path_short(self, obj):
        return (obj.background_image_path[:50] + '…') if obj.background_image_path and len(obj.background_image_path) > 50 else (obj.background_image_path or '—')
    background_image_path_short.short_description = 'Plik wzorca'


@admin.register(Dataset)
class DatasetAdmin(admin.ModelAdmin):
    list_display = ['name', 'images_count', 'labels_count', 'created_at']
    list_filter = ['created_at']
    search_fields = ['name', 'description']


@admin.register(DatasetImage)
class DatasetImageAdmin(admin.ModelAdmin):
    list_display = ['dataset', 'image_path', 'has_label', 'captured_at']
    list_filter = ['has_label', 'captured_at']
    search_fields = ['image_path']


@admin.register(ModelTraining)
class ModelTrainingAdmin(admin.ModelAdmin):
    list_display = ['name', 'base_model', 'epochs', 'status', 'final_map50', 'created_at']
    list_filter = ['status', 'base_model', 'created_at']
    search_fields = ['name']


@admin.register(DetectionLog)
class DetectionLogAdmin(admin.ModelAdmin):
    list_display = ['camera', 'model', 'detection_count', 'status', 'detected_at']
    list_filter = ['status', 'detected_at']
    search_fields = ['camera__name', 'status']
