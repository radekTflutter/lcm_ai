from django.urls import re_path
from factory import consumers

websocket_urlpatterns = [
    re_path(r'ws/camera/(?P<camera_id>\w+)/$', consumers.CameraConsumer.as_asgi()),
    re_path(r'ws/training/(?P<training_id>\w+)/$', consumers.TrainingConsumer.as_asgi()),
    re_path(r'ws/detection/$', consumers.DetectionConsumer.as_asgi()),
]
