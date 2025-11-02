"""
Frame Processing Handler
Async queue-based frame processing to prevent blocking
"""

import threading
import queue
import time
import logging
from typing import Optional, Callable
import numpy as np

logger = logging.getLogger(__name__)


class FrameProcessor:
    """
    Asynchronous frame processor with queue management
    Prevents blocking and automatically drops old frames
    """

    def __init__(
        self, process_func: Callable, max_queue_size: int = 2, worker_threads: int = 1
    ):
        """
        Initialize frame processor

        Args:
            process_func: Function to process frames (takes image, returns result)
            max_queue_size: Maximum frames in queue (older frames dropped)
            worker_threads: Number of worker threads
        """
        self.process_func = process_func
        self.max_queue_size = max_queue_size
        self.worker_threads = worker_threads

        # Queue for incoming frames
        self.frame_queue = queue.Queue(maxsize=max_queue_size)

        # Result storage (latest result only)
        self.latest_result = None
        self.result_lock = threading.Lock()

        # Statistics
        self.frames_processed = 0
        self.frames_dropped = 0
        self.total_processing_time = 0.0

        # Control
        self.running = False
        self.workers = []

        logger.info(
            f"FrameProcessor initialized: {worker_threads} workers, queue size {max_queue_size}"
        )

    def start(self):
        """Start worker threads"""
        if self.running:
            logger.warning("FrameProcessor already running")
            return

        self.running = True

        # Start worker threads
        for i in range(self.worker_threads):
            worker = threading.Thread(
                target=self._worker_loop, name=f"FrameWorker-{i}", daemon=True
            )
            worker.start()
            self.workers.append(worker)

        logger.info(f"Started {len(self.workers)} worker threads")

    def stop(self):
        """Stop worker threads gracefully"""
        if not self.running:
            return

        logger.info("Stopping frame processor...")
        self.running = False

        # Clear queue
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break

        # Wait for workers
        for worker in self.workers:
            worker.join(timeout=2.0)

        self.workers.clear()
        logger.info("Frame processor stopped")

    def submit_frame(self, image: np.ndarray, metadata: dict = None) -> bool:
        """
        Submit frame for processing

        Args:
            image: Frame as numpy array
            metadata: Optional metadata dict

        Returns:
            True if frame queued, False if dropped
        """
        if not self.running:
            logger.warning("Cannot submit frame: processor not running")
            return False

        frame_data = {
            "image": image,
            "metadata": metadata or {},
            "submit_time": time.time(),
        }

        try:
            # Try to put in queue without blocking
            self.frame_queue.put_nowait(frame_data)
            return True
        except queue.Full:
            # Queue full - drop oldest frame and try again
            try:
                self.frame_queue.get_nowait()  # Drop oldest
                self.frames_dropped += 1
                self.frame_queue.put_nowait(frame_data)
                logger.debug(
                    f"Dropped old frame (queue full), total dropped: {self.frames_dropped}"
                )
                return True
            except:
                self.frames_dropped += 1
                logger.warning("Failed to queue frame")
                return False

    def get_latest_result(self):
        """
        Get latest processing result

        Returns:
            Latest result dict or None
        """
        with self.result_lock:
            return self.latest_result

    def _worker_loop(self):
        """Worker thread main loop"""
        logger.info(f"Worker {threading.current_thread().name} started")

        while self.running:
            try:
                # Get frame from queue with timeout
                frame_data = self.frame_queue.get(timeout=0.1)

                # Process frame
                start_time = time.time()

                try:
                    result = self.process_func(frame_data["image"])
                    processing_time = (time.time() - start_time) * 1000  # ms

                    # Store result
                    with self.result_lock:
                        self.latest_result = {
                            "result": result,
                            "processing_time_ms": processing_time,
                            "timestamp": time.time(),
                            "metadata": frame_data["metadata"],
                        }

                    # Update stats
                    self.frames_processed += 1
                    self.total_processing_time += processing_time

                    if self.frames_processed % 30 == 0:  # Log every 30 frames
                        avg_time = self.total_processing_time / self.frames_processed
                        logger.info(
                            f"Processed {self.frames_processed} frames, "
                            f"avg: {avg_time:.1f}ms, dropped: {self.frames_dropped}"
                        )

                except Exception as e:
                    logger.error(f"Error processing frame: {e}", exc_info=True)

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Worker error: {e}", exc_info=True)

        logger.info(f"Worker {threading.current_thread().name} stopped")

    def get_stats(self) -> dict:
        """Get processing statistics"""
        avg_time = 0.0
        if self.frames_processed > 0:
            avg_time = self.total_processing_time / self.frames_processed

        return {
            "frames_processed": self.frames_processed,
            "frames_dropped": self.frames_dropped,
            "avg_processing_time_ms": round(avg_time, 2),
            "queue_size": self.frame_queue.qsize(),
            "running": self.running,
        }
