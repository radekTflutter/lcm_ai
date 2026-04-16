"""
URL configuration for factory app
"""
from django.urls import path
from . import views

urlpatterns = [
    path('', views.wizard_view, name='wizard'),
    path('wizard/<int:stage>/', views.wizard_view, name='wizard_stage'),
    path('api/system/status/', views.get_system_status, name='get_system_status'),
    path('api/camera/create/', views.create_camera, name='create_camera'),
    path('api/camera/<int:camera_id>/connect/', views.connect_camera, name='connect_camera'),
    path('api/camera/<int:camera_id>/disconnect/', views.disconnect_camera, name='disconnect_camera'),
    path('api/camera/<int:camera_id>/frame/', views.get_camera_frame, name='get_camera_frame'),
    path('api/camera/<int:camera_id>/url/', views.get_camera_debug_url, name='get_camera_debug_url'),
    path('api/camera/<int:camera_id>/roi/', views.set_camera_roi, name='set_camera_roi'),
    path('api/camera/<int:camera_id>/roi_status/', views.get_camera_roi_status, name='get_camera_roi_status'),
    path('api/calibration/<int:camera_id>/capture/', views.capture_background, name='capture_background'),
    path('api/calibration/<int:camera_id>/use_admin/', views.use_admin_background, name='use_admin_background'),
    path('api/calibration/<int:camera_id>/calculate/', views.calculate_hsv, name='calculate_hsv'),
    path('api/dataset/create/', views.create_dataset, name='create_dataset'),
    path('api/dataset/<int:dataset_id>/stats/', views.get_dataset_stats, name='get_dataset_stats'),
    path('api/dataset/<int:dataset_id>/start_collection/', views.start_data_collection, name='start_data_collection'),
    path('api/dataset/<int:dataset_id>/stop_collection/', views.stop_data_collection, name='stop_data_collection'),
    path('api/dataset/<int:dataset_id>/add_labeled_image/', views.add_labeled_image, name='add_labeled_image'),
    path('api/training/create/', views.create_training, name='create_training'),
    path('api/training/<int:training_id>/start/', views.start_training, name='start_training'),
    path('api/training/<int:training_id>/status/', views.get_training_status, name='get_training_status'),
    path('api/detection/start/', views.start_detection, name='start_detection'),
    path('api/detection/stop/', views.stop_detection, name='stop_detection'),
    path('api/detection/status/', views.get_detection_status, name='get_detection_status'),
]
