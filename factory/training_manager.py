"""
Training manager for YOLO models with CUDA support.
torch and ultralytics are imported only when needed, so the app runs without them.
"""
import logging
import threading
from pathlib import Path
from typing import Optional, Callable

from django.conf import settings

logger = logging.getLogger(__name__)

# Cache żeby nie importować torch przy każdym zapytaniu (torch import trwa 2–10 s)
_cuda_available_cache: Optional[bool] = None
_available_memory_cache: Optional[int] = None


def check_cuda_available() -> bool:
    """Check if CUDA is available. Wynik cache'owany (import torch tylko raz)."""
    global _cuda_available_cache
    if _cuda_available_cache is not None:
        return _cuda_available_cache
    try:
        import torch
        _cuda_available_cache = bool(torch.cuda.is_available())
    except Exception:
        _cuda_available_cache = False
    return _cuda_available_cache


def get_available_memory_mb() -> Optional[int]:
    """Get available memory in MB (for Jetson). Cache'owane."""
    global _available_memory_cache
    if _available_memory_cache is not None:
        return _available_memory_cache
    try:
        import psutil
        _available_memory_cache = int(psutil.virtual_memory().available / (1024 * 1024))
    except Exception:
        _available_memory_cache = None
    return _available_memory_cache


class TrainingManager:
    """Manages YOLO model training with progress tracking"""
    
    def __init__(self, training_id: int, callback: Optional[Callable] = None):
        self.training_id = training_id
        self.callback = callback
        self.is_running = False
        self.training_thread = None
        self.model = None
        self.device = 'cuda' if check_cuda_available() else 'cpu'
        logger.info("TrainingManager initialized for training %s, device: %s", training_id, self.device)
    
    def start_training(
        self,
        dataset_path: str,
        base_model: str = 'yolov8n.pt',
        epochs: int = 100,
        batch_size: int = 16,
        image_size: int = 640,
        project_dir: Optional[str] = None
    ):
        """Start training in background thread"""
        if self.is_running:
            logger.warning("Training already running")
            return
        
        self.is_running = True
        self.training_thread = threading.Thread(
            target=self._train_model,
            args=(dataset_path, base_model, epochs, batch_size, image_size, project_dir),
            daemon=True
        )
        self.training_thread.start()
    
    def _train_model(
        self,
        dataset_path: str,
        base_model: str,
        epochs: int,
        batch_size: int,
        image_size: int,
        project_dir: Optional[str]
    ):
        """Internal training method (imports ultralytics/torch only here)."""
        from django.utils import timezone
        from ultralytics import YOLO

        try:
            from factory.models import ModelTraining

            training = ModelTraining.objects.get(id=self.training_id)
            training.status = 'running'
            training.started_at = timezone.now()
            training.save()

            if self.callback:
                try:
                    self.callback({'type': 'started', 'message': 'Trening w toku...'})
                except Exception as cb_err:
                    logger.warning("Training callback failed: %s", cb_err)

            # Check available memory
            available_mem = get_available_memory_mb()
            if available_mem and available_mem < 2000:  # Less than 2GB
                logger.warning("Low available memory: %s MB", available_mem)
            if self.callback:
                try:
                    self.callback({
                        'type': 'warning',
                        'message': f'Low available memory: {available_mem:.0f} MB. Consider closing other applications.'
                    })
                except Exception:
                    pass

            # Prepare dataset structure (YOLO format)
            dataset_yaml = self._prepare_dataset_yaml(dataset_path)
            self._ensure_val_set(dataset_path)

            # Set project directory
            if project_dir is None:
                project_dir = str(Path(settings.MODELS_ROOT) / f"training_{self.training_id}")

            # Load model
            logger.info("Loading model: %s", base_model)
            self.model = YOLO(base_model)

            # Move to CUDA if available
            if self.device == 'cuda':
                logger.info("Using CUDA for training")
                if hasattr(self.model.model, 'to'):
                    self.model.model.to(self.device)

            # Progress callback: send epoch/progress to UI
            def _on_epoch_end(trainer):
                try:
                    epoch_one = getattr(trainer, 'epoch', 0) + 1
                    total = getattr(trainer, 'epochs', epochs) or epochs
                    progress = int(round(epoch_one / total * 100)) if total else 0
                    loss_val = None
                    if getattr(trainer, 'loss', None) is not None:
                        t = trainer.loss
                        loss_val = float(t) if hasattr(t, 'item') else float(t)
                    elif getattr(trainer, 'tloss', None) is not None:
                        t = trainer.tloss
                        loss_val = float(t) if hasattr(t, 'item') else float(t)
                    metrics = getattr(trainer, 'metrics', None) or {}
                    map50 = metrics.get('metrics/mAP50(B)', metrics.get('mAP50'))
                    map50_95 = metrics.get('metrics/mAP50-95(B)', metrics.get('mAP50-95'))
                    if map50 is not None and hasattr(map50, 'item'):
                        map50 = float(map50)
                    if map50_95 is not None and hasattr(map50_95, 'item'):
                        map50_95 = float(map50_95)
                    payload = {
                        'type': 'update',
                        'epoch': epoch_one,
                        'progress': min(progress, 100),
                        'message': f'Epoch {epoch_one}/{total}',
                    }
                    if loss_val is not None:
                        payload['loss'] = loss_val
                    if map50 is not None:
                        payload['map50'] = map50
                    if map50_95 is not None:
                        payload['map50_95'] = map50_95
                    if self.callback:
                        self.callback(payload)
                    # Update DB so polling GET status also gets progress
                    try:
                        ModelTraining.objects.filter(id=self.training_id).update(
                            current_epoch=payload.get('epoch'),
                            progress_percent=payload.get('progress')
                        )
                    except Exception:
                        pass
                except Exception as e:
                    logger.warning("Epoch callback error: %s", e)

            self.model.add_callback("on_train_epoch_end", _on_epoch_end)
            logger.info("Starting training: epochs=%s, batch=%s, size=%s", epochs, batch_size, image_size)

            results = self.model.train(
                data=dataset_yaml,
                epochs=epochs,
                batch=batch_size,
                imgsz=image_size,
                device=self.device,
                project=project_dir,
                name='run',
                exist_ok=True,
                verbose=True
            )
            
            # Extract results
            metrics = results.results_dict if hasattr(results, 'results_dict') else {}
            
            # Find best model
            best_model_path = Path(project_dir) / 'run' / 'weights' / 'best.pt'
            if not best_model_path.exists():
                # Try alternative location
                best_model_path = Path(project_dir) / 'weights' / 'best.pt'
            
            # Update training record
            training.status = 'completed'
            training.completed_at = timezone.now()
            training.best_model_path = str(best_model_path) if best_model_path.exists() else None
            training.final_map50 = metrics.get('metrics/mAP50(B)', metrics.get('mAP50', None))
            training.final_map50_95 = metrics.get('metrics/mAP50-95(B)', metrics.get('mAP50-95', None))
            training.training_loss = metrics.get('train/box_loss', None)
            training.validation_loss = metrics.get('val/box_loss', None)
            training.save()

            logger.info("Training completed: mAP50=%s", training.final_map50)

            if self.callback:
                try:
                    self.callback({
                        'type': 'completed',
                        'map50': training.final_map50,
                        'map50_95': training.final_map50_95,
                        'best_model_path': training.best_model_path
                    })
                except Exception:
                    pass
            
        except Exception as e:
            logger.error(f"Training error: {e}", exc_info=True)
            
            from factory.models import ModelTraining
            try:
                training = ModelTraining.objects.get(id=self.training_id)
                training.status = 'failed'
                training.save()
            except:
                pass
            
            if self.callback:
                try:
                    self.callback({
                        'type': 'error',
                        'message': str(e)
                    })
                except Exception:
                    pass
        
        finally:
            self.is_running = False
    
    def _prepare_dataset_yaml(self, dataset_path: str) -> str:
        """Prepare dataset.yaml file for YOLO training"""
        dataset_dir = Path(dataset_path)
        
        # Create dataset.yaml
        yaml_path = dataset_dir / 'dataset.yaml'
        
        yaml_content = f"""# YOLO Dataset Configuration
path: {dataset_dir.absolute()}
train: images/train
val: images/val
test: images/test

# Classes
names:
  0: can
"""
        
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        
        logger.info("Created dataset.yaml: %s", yaml_path)
        return str(yaml_path)

    def _ensure_val_set(self, dataset_path: str) -> None:
        """If images/val is empty, copy a subset of train images (and labels) to val so YOLO can run."""
        import shutil
        dataset_dir = Path(dataset_path)
        train_img = dataset_dir / "images" / "train"
        val_img = dataset_dir / "images" / "val"
        train_lbl = dataset_dir / "labels" / "train"
        val_lbl = dataset_dir / "labels" / "val"
        val_img.mkdir(parents=True, exist_ok=True)
        val_lbl.mkdir(parents=True, exist_ok=True)
        val_images = list(val_img.glob("*.*")) if val_img.exists() else []
        # Skip if val already has images
        if val_images:
            return
        train_images = list(train_img.glob("*.jpg")) + list(train_img.glob("*.jpeg")) + list(train_img.glob("*.png"))
        if not train_images:
            return
        # Use ~10% for val, at least 1, at most 100
        n_val = min(max(1, len(train_images) // 10), 100, len(train_images))
        import random
        chosen = random.sample(train_images, n_val)
        for img_path in chosen:
            dst_img = val_img / img_path.name
            if not dst_img.exists():
                shutil.copy2(img_path, dst_img)
            stem = img_path.stem
            lbl_src = train_lbl / f"{stem}.txt"
            if lbl_src.exists():
                lbl_dst = val_lbl / f"{stem}.txt"
                if not lbl_dst.exists():
                    shutil.copy2(lbl_src, lbl_dst)
        logger.info("Ensured val set: %s images in val", n_val)

    def stop_training(self):
        """Stop training (if possible)."""
        self.is_running = False
        logger.info("Training stop requested")
