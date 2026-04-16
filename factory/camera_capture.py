"""
Camera capture module for Axis cameras.
Obsługuje OpenCV oraz fallback przez requests + Digest auth (VAPIX zaleca Digest dla HTTP).
"""
import cv2
import numpy as np
import threading
import time
import re
from typing import Optional, Callable
import logging

logger = logging.getLogger(__name__)

# Stałe JPEG (start/koniec ramki)
JPEG_SOI = b'\xff\xd8'
JPEG_EOI = b'\xff\xd9'


def _url_without_credentials(url: str) -> str:
    """Zwraca URL bez credentials w środku (scheme + host + path)."""
    if '@' not in url:
        return url.strip()
    try:
        before_at = url.split('@', 1)[0]
        after_at = url.split('@', 1)[1]
        if '://' in before_at:
            scheme_rest = before_at.split('://', 1)[1]
            if ':' in scheme_rest:
                # user:pass
                return before_at.split(':', 2)[0] + '://' + after_at
        return 'http://' + after_at
    except Exception:
        return url.strip()


class CameraCapture:
    """Thread-safe camera capture class for Axis cameras. OpenCV lub requests+Digest (VAPIX)."""

    def __init__(
        self,
        camera_ip: str,
        username: str = None,
        password: str = None,
        port: int = 80,
        stream_url: str = None,
    ):
        self.camera_ip = camera_ip
        self.username = (username or "").strip()
        self.password = (password or "").strip()
        self.port = port
        self.stream_url = (stream_url or "").strip()
        self.cap = None
        self.is_running = False
        self.current_frame = None
        self.lock = threading.Lock()
        self.frame_callback: Optional[Callable] = None
        self.capture_thread = None
        self._use_requests_stream = False
        self._stream_response = None

    def _build_http_url(self, with_credentials: bool = False) -> str:
        """URL MJPEG Axis: http://IP/axis-cgi/mjpg/video.cgi (z creds tylko dla Basic/OpenCV)."""
        base = f"http://{self.camera_ip}/axis-cgi/mjpg/video.cgi"
        if with_credentials and self.username and self.password:
            return f"http://{self.username}:{self.password}@{self.camera_ip}/axis-cgi/mjpg/video.cgi"
        return base

    def _get_stream_url_for_requests(self) -> str:
        """URL do requesta (bez credentials – Digest przekazujemy w auth=)."""
        if self.stream_url:
            return _url_without_credentials(self.stream_url)
        return self._build_http_url(with_credentials=False)

    def _connect_axis_requests(self, url: str) -> bool:
        """Połączenie przez requests + Digest auth (VAPIX). Uruchamia wątek parsujący MJPEG."""
        try:
            import requests
            from requests.auth import HTTPDigestAuth
        except ImportError:
            logger.warning("requests nie zainstalowany – brak fallback Digest auth. pip install requests")
            return False

        auth = None
        if self.username or self.password:
            auth = HTTPDigestAuth(self.username, self.password)

        try:
            resp = requests.get(
                url,
                auth=auth,
                stream=True,
                timeout=(10, 30),
                headers={"User-Agent": "Axis-VAPIX-MJPEG-Client/1.0"},
            )
            resp.raise_for_status()
        except Exception as e:
            logger.warning("requests stream (Digest) failed: %s", e)
            return False

        self._stream_response = resp
        self._use_requests_stream = True
        self.cap = None
        self.is_running = True
        self.capture_thread = threading.Thread(target=self._capture_loop_mjpeg_stream, daemon=True)
        self.capture_thread.start()
        logger.info("Camera connected via requests+Digest (VAPIX)")
        return True

    def _capture_loop_mjpeg_stream(self):
        """Wątek: odczyt strumienia multipart/x-mixed-replace i wyciąganie ramek JPEG."""
        if not self._stream_response:
            return
        buffer = b""
        try:
            for chunk in self._stream_response.iter_content(chunk_size=8192):
                if not self.is_running:
                    break
                buffer += chunk
                while True:
                    a = buffer.find(JPEG_SOI)
                    b = buffer.find(JPEG_EOI, a) if a != -1 else -1
                    if a == -1 or b == -1:
                        if len(buffer) > 2 * 1024 * 1024:
                            buffer = buffer[-1024:]
                        break
                    jpg = buffer[a:b + 2]
                    buffer = buffer[b + 2:]
                    try:
                        arr = np.frombuffer(jpg, dtype=np.uint8)
                        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                        if frame is not None:
                            with self.lock:
                                self.current_frame = frame
                            if self.frame_callback:
                                self.frame_callback(frame)
                    except Exception as e:
                        logger.debug("imdecode error: %s", e)
        except Exception as e:
            if self.is_running:
                logger.error("MJPEG stream read error: %s", e)
        finally:
            try:
                self._stream_response.close()
            except Exception:
                pass
            self._stream_response = None

    def connect(self) -> bool:
        """Łączy z kamerą: najpierw OpenCV, przy niepowodzeniu – requests + Digest (VAPIX)."""
        url = self.stream_url if self.stream_url else self._build_http_url(with_credentials=True)
        try:
            try:
                self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
            except (TypeError, AttributeError):
                self.cap = cv2.VideoCapture(url)
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                try:
                    self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)
                    self.cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
                except (AttributeError, cv2.error):
                    pass
                logger.info("Camera connected via OpenCV: %s", url[:80])
                return True
            self.cap.release()
            self.cap = None
        except Exception as e:
            logger.warning("OpenCV connect failed: %s", e)
            self.cap = None

        url_clean = self._get_stream_url_for_requests()
        logger.info("Trying Axis VAPIX (requests+Digest): %s", url_clean[:80])
        return self._connect_axis_requests(url_clean)
    
    def start_capture(self, callback: Optional[Callable] = None):
        """Start continuous frame capture (OpenCV) lub ustaw callback (gdy stream przez requests)."""
        if self._use_requests_stream:
            self.frame_callback = callback
            logger.info("Camera capture (requests stream) callback set")
            return
        if self.is_running:
            logger.warning("Capture already running")
            return
        if not self.cap or not self.cap.isOpened():
            if not self.connect():
                raise RuntimeError("Failed to connect to camera")
        self.is_running = True
        self.frame_callback = callback
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        logger.info("Camera capture started")
    
    def _capture_loop(self):
        """Internal capture loop running in background thread (OpenCV)."""
        while self.is_running and self.cap and self.cap.isOpened():
            try:
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    with self.lock:
                        self.current_frame = frame.copy()
                    if self.frame_callback:
                        self.frame_callback(frame)
                else:
                    time.sleep(0.1)
            except Exception as e:
                logger.debug("Capture loop read error (ignored): %s", e)
                time.sleep(0.1)
    
    def get_frame(self) -> Optional[np.ndarray]:
        """Get current frame (thread-safe)"""
        with self.lock:
            if self.current_frame is not None:
                return self.current_frame.copy()
        return None
    
    def capture_single(self) -> Optional[np.ndarray]:
        """Capture a single frame (OpenCV lub ostatnia ramka ze streamu requests). Nie rzuca przy błędzie MJPEG/OpenCV."""
        if self._use_requests_stream:
            for _ in range(50):
                f = self.get_frame()
                if f is not None:
                    return f
                time.sleep(0.1)
            return self.get_frame()
        if not self.cap or not self.cap.isOpened():
            if not self.connect():
                return None
        try:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                return frame
        except Exception as e:
            logger.debug("capture_single read error: %s", e)
        return None

    def stop_capture(self):
        """Stop continuous capture"""
        self.is_running = False
        if self._stream_response:
            try:
                self._stream_response.close()
            except Exception:
                pass
            self._stream_response = None
        if self.capture_thread:
            self.capture_thread.join(timeout=3.0)
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        self._use_requests_stream = False
        logger.info("Camera capture stopped")
    
    def release(self):
        """Release camera resources"""
        self.stop_capture()
    
    def __enter__(self):
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


# Global camera instances storage
_camera_instances = {}
_camera_locks = {}


def get_camera_instance(camera_id: int) -> Optional[CameraCapture]:
    """Get or create camera instance for given camera ID (uses Camera.get_http_url())."""
    if camera_id not in _camera_instances:
        from factory.models import Camera
        try:
            camera = Camera.objects.get(id=camera_id)
            stream_url = camera.get_http_url()
            _camera_instances[camera_id] = CameraCapture(
                camera_ip=camera.ip_address,
                username=camera.username or "",
                password=camera.password or "",
                port=camera.port or 80,
                stream_url=stream_url,
            )
            _camera_locks[camera_id] = threading.Lock()
        except Camera.DoesNotExist:
            logger.warning("Camera id=%s does not exist", camera_id)
            return None
        except Exception as e:
            logger.exception("Failed to create camera instance for id=%s: %s", camera_id, e)
            return None

    return _camera_instances.get(camera_id)


def release_camera_instance(camera_id: int):
    """Release camera instance. Nigdy nie rzuca wyjątku."""
    if camera_id not in _camera_instances:
        return
    try:
        _camera_instances[camera_id].release()
    except Exception as e:
        logger.exception("release_camera_instance(id=%s): %s", camera_id, e)
    finally:
        if camera_id in _camera_instances:
            del _camera_instances[camera_id]
        if camera_id in _camera_locks:
            del _camera_locks[camera_id]
