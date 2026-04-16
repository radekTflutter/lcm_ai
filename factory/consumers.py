"""
WebSocket consumers for real-time updates
"""
import json
import base64
import cv2
import numpy as np
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.db import database_sync_to_async
from factory.camera_capture import get_camera_instance
from factory.models import Camera, ModelTraining
from django.db import DatabaseError
import logging

logger = logging.getLogger(__name__)


class CameraConsumer(AsyncWebsocketConsumer):
    """WebSocket consumer for camera live feed"""

    async def connect(self):
        try:
            self.camera_id = self.scope['url_route']['kwargs']['camera_id']
            self.room_group_name = f'camera_{self.camera_id}'
            await self.accept()
            await self.channel_layer.group_add(
                self.room_group_name,
                self.channel_name
            )
            logger.info("Camera WebSocket connected: %s", self.camera_id)
        except Exception as e:
            logger.exception("WebSocket connect failed: %s", e)
            try:
                await self.accept()
            except Exception:
                pass
            try:
                await self.close(code=1011)
            except Exception:
                pass

    async def disconnect(self, close_code):
        try:
            await self.channel_layer.group_discard(
                self.room_group_name,
                self.channel_name
            )
        except Exception as e:
            logger.warning("disconnect group_discard: %s", e)
        logger.info("Camera WebSocket disconnected: %s", self.camera_id)

    async def receive(self, text_data):
        try:
            data = json.loads(text_data)
            if data.get('type') == 'get_frame':
                await self.send_frame()
        except Exception as e:
            logger.exception("WebSocket receive: %s", e)
            try:
                await self.send(text_data=json.dumps({'type': 'error', 'message': str(e)[:200]}))
            except Exception:
                pass
    
    async def send_frame(self):
        """Send current frame to client. On error send type 'error' so frontend can show message."""
        try:
            await database_sync_to_async(Camera.objects.get)(id=int(self.camera_id))
        except Camera.DoesNotExist:
            logger.warning("Camera id=%s not found for WebSocket frame", self.camera_id)
            await self.send(text_data=json.dumps({
                'type': 'error',
                'message': 'Kamera nie istnieje.'
            }))
            return
        except (DatabaseError, ValueError) as e:
            logger.exception("Camera lookup error for id=%s: %s", self.camera_id, e)
            await self.send(text_data=json.dumps({
                'type': 'error',
                'message': 'Błąd bazy danych.'
            }))
            return

        try:
            camera_instance = await database_sync_to_async(get_camera_instance)(int(self.camera_id))
        except Exception as e:
            logger.exception("get_camera_instance failed for id=%s: %s", self.camera_id, e)
            await self.send(text_data=json.dumps({
                'type': 'error',
                'message': 'Nie można utworzyć połączenia z kamerą.'
            }))
            return

        if not camera_instance:
            await self.send(text_data=json.dumps({
                'type': 'error',
                'message': 'Kamera nie jest skonfigurowana.'
            }))
            return

        try:
            frame = camera_instance.get_frame()
            if frame is None:
                frame = camera_instance.capture_single()
            if frame is not None:
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                frame_base64 = base64.b64encode(buffer).decode('utf-8')
                await self.send(text_data=json.dumps({
                    'type': 'frame',
                    'image': f'data:image/jpeg;base64,{frame_base64}'
                }))
            else:
                await self.send(text_data=json.dumps({
                    'type': 'error',
                    'message': 'Brak sygnału z kamery (sprawdź IP, zasilanie i sieć).'
                }))
        except Exception as e:
            logger.exception("Error encoding/sending frame for camera id=%s: %s", self.camera_id, e)
            await self.send(text_data=json.dumps({
                'type': 'error',
                'message': f'Błąd odczytu klatki: {str(e)}'
            }))
    
    async def camera_frame(self, event):
        """Receive frame from channel layer"""
        await self.send(text_data=json.dumps(event['data']))


class TrainingConsumer(AsyncWebsocketConsumer):
    """WebSocket consumer for training progress"""
    
    async def connect(self):
        self.training_id = self.scope['url_route']['kwargs']['training_id']
        self.room_group_name = f'training_{self.training_id}'
        
        await self.channel_layer.group_add(
            self.room_group_name,
            self.channel_name
        )
        
        await self.accept()
        logger.info(f"Training WebSocket connected: {self.training_id}")
    
    async def disconnect(self, close_code):
        await self.channel_layer.group_discard(
            self.room_group_name,
            self.channel_name
        )
        logger.info(f"Training WebSocket disconnected: {self.training_id}")
    
    async def receive(self, text_data):
        pass
    
    async def training_update(self, event):
        """Receive training update from channel layer"""
        await self.send(text_data=json.dumps(event['data']))


class DetectionConsumer(AsyncWebsocketConsumer):
    """WebSocket consumer for live detection"""
    
    async def connect(self):
        self.room_group_name = 'detection'
        
        await self.channel_layer.group_add(
            self.room_group_name,
            self.channel_name
        )
        
        await self.accept()
        logger.info("Detection WebSocket connected")
    
    async def disconnect(self, close_code):
        await self.channel_layer.group_discard(
            self.room_group_name,
            self.channel_name
        )
        logger.info("Detection WebSocket disconnected")
    
    async def receive(self, text_data):
        pass
    
    async def detection_update(self, event):
        """Receive detection update from channel layer"""
        await self.send(text_data=json.dumps(event['data']))
