from .client import APIError, Laminar
from .tracing_types import Span, Trace

from queue import Queue, Empty, Full
from typing import Union

import atexit
import backoff
import logging
import time
import threading


class Collector(threading.Thread):
    _log = logging.getLogger("laminar.collector")
    _queue: Queue[Union[Span, Trace]]
    _client: Laminar
    _flush_interval: float

    def __init__(
        self,
        queue: Queue[Union[Span, Trace]],
        client: Laminar,
        flush_interval: float = 5.0,
    ):
        super().__init__()
        self.daemon = True
        self._queue = queue
        self.running = True
        self._flush_interval = flush_interval
        self._client = client

    def run(self):
        """Runs the collector."""
        self._log.debug("collector is running...")
        while self.running:
            self.upload()

    def upload(self):
        """Upload the next batch of items, return whether successful."""
        batch = self._next()
        if len(batch) == 0:
            return

        try:
            self._upload_batch(batch)
        except Exception as e:
            self._log.exception("error uploading: %s", e)
        finally:
            # mark items as acknowledged from queue
            for _ in batch:
                self._queue.task_done()

    def pause(self):
        self.running = False

    def _next(self):
        items = []
        start_time = time.monotonic()

        while True:
            elapsed = time.monotonic() - start_time
            if elapsed >= self._flush_interval:
                break
            try:
                item = self._queue.get(
                    block=True, timeout=self._flush_interval - elapsed
                )
                items.append(item)
            except Empty:
                break
        return items

    def _upload_batch(self, batch: list[Union[Trace, Span]]):
        self._log.debug("uploading batch of %d items", len(batch))

        @backoff.on_exception(backoff.expo, Exception, max_tries=5)
        def execute_task_with_backoff(batch: list[Union[Trace, Span]]):
            try:
                self._client.batch_post_traces(batch=batch)
            except Exception as e:
                if (
                    isinstance(e, APIError)
                    and 400 <= int(e.status) < 500
                    and int(e.status) != 429
                ):
                    self._log.warn(
                        f"Received {e.status} error by Laminar server, not retrying: {e.message}"
                    )
                    return

                raise e

        execute_task_with_backoff(batch)
        self._log.debug("successfully uploaded batch of %d items", len(batch))


class ThreadManager:
    _log = logging.getLogger("laminar.task_manager")
    _queue: Queue[Union[Span, Trace]]
    _client: Laminar
    _max_task_queue_size: int
    _flush_interval: float
    _collectors: list[Collector] = []
    _threads: int

    def __init__(
        self,
        client: Laminar,
        flush_interval: float = 2.0,
        max_task_queue_size: int = 1000,
        threads: int = 1,
    ):
        self._max_task_queue_size = max_task_queue_size
        self._threads = threads
        self._queue = Queue(maxsize=self._max_task_queue_size)
        self._flush_interval = flush_interval
        self._client = client
        for _ in range(self._threads):
            collector = Collector(
                queue=self._queue,
                client=self._client,
                flush_interval=flush_interval,
            )
            self._collectors.append(collector)
            collector.start()
        atexit.register(self.join)

    def add_task(self, event: Union[Span, Trace]) -> bool:
        try:
            self._queue.put(event, block=False)
            return True
        except Full:
            self._log.warning("queue is full")
            return False
        except Exception as e:
            self._log.exception(f"Exception in adding task {e}")

            return False

    def flush(self):
        """Forces a flush from the internal queue to the server"""
        self._log.debug("flushing queue")
        queue = self._queue
        size = queue.qsize()
        queue.join()
        # Note that this message may not be precise, because of threading.
        self._log.debug("successfully flushed about %s items.", size)

    def join(self):
        """Ends the collector threads once the queue is empty.
        Blocks execution until finished
        """
        self._log.debug(f"joining {len(self._collectors)} collector threads")

        # pause all collectors before joining them so we don't have to wait for multiple
        # flush intervals to join them all.
        for collector in self._collectors:
            collector.pause()

        for i, collector in enumerate(self._collectors):
            try:
                collector.join()
            except RuntimeError:
                # collector thread has not started
                pass

            self._log.debug(f"collector thread {i} joined")

    def shutdown(self):
        """Flush all messages and cleanly shutdown the client"""
        self._log.debug("shutdown initiated")

        self.flush()
        self.join()

        self._log.debug("shutdown completed")
