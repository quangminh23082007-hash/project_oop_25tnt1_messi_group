"""Bounded, ordered embedding workers with explicit failure/timeout handling.

Each spawned worker loads its own encoder on its assigned device. The parent
never owns the language model and does not share its weights through /dev/shm.
Only small text chunks and NumPy embedding batches cross process boundaries.
"""
from __future__ import annotations

import math
import multiprocessing as mp
from queue import Empty, Full, Queue
import threading
import time
import traceback

import numpy as np

from .models import positive_int


def load_qwen_worker(device, config):
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import torch
    torch.set_num_threads(1)
    from .text_features import QwenSentenceEncoder
    return QwenSentenceEncoder(device=device, **config)


def _worker_main(index, device, config, batch_size, inbox, outbox, factory):
    encoder = None
    try:
        encoder = factory(device, config)
        outbox.put(("ready", index, int(encoder.dimension), dict(encoder.info)))
        while True:
            task = inbox.get()
            if task is None:
                return
            job_id, texts = task
            parts = []
            for start in range(0, len(texts), batch_size):
                batch = texts[start:start + batch_size]
                values = np.asarray(encoder.encode(batch), dtype=np.float32)
                if values.shape != (len(batch), encoder.dimension) or not np.isfinite(values).all():
                    raise ValueError("Worker encoder returned invalid embeddings")
                parts.append(values)
            outbox.put(("result", index, job_id, np.concatenate(parts, axis=0)))
    except BaseException:
        outbox.put(("error", index, traceback.format_exc()))
    finally:
        if encoder is not None and callable(getattr(encoder, "close", None)):
            encoder.close()


class ProcessEncoderPool:
    """Persistent independent encoders; encode() preserves input order.

    At most one chunk is in flight per worker. `batch_size` is the worker GPU
    microbatch, while `chunk_size` bounds the texts sent to that worker at once.
    The factory argument exists for offline CPU integration tests.
    """

    def __init__(self, devices, config, *, batch_size=2, chunk_size=64,
                 timeout=600, factory=load_qwen_worker):
        if not devices:
            raise ValueError("At least one worker device is required")
        self.devices = list(devices)
        self.batch_size = positive_int(batch_size, "batch_size")
        self.chunk_size = positive_int(chunk_size, "chunk_size")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("worker timeout must be finite and positive")
        self.timeout = float(timeout)
        self.dispatch_size = self.chunk_size * len(self.devices)
        self.processed_counts = [0] * len(self.devices)
        self.closed, self._job_id = False, 0
        context = mp.get_context("spawn")
        self._outbox = context.Queue(maxsize=2 * len(self.devices))
        self._messages = Queue()
        self._receiver_stop = threading.Event()
        self._receiver = threading.Thread(target=self._collect_messages, daemon=True)
        self._receiver.start()
        self._inboxes = [context.Queue(maxsize=1) for _ in self.devices]
        self.processes = []
        self.worker_infos = []
        try:
            for index, device in enumerate(self.devices):
                process = context.Process(target=_worker_main,
                    args=(index, device, dict(config), self.batch_size,
                          self._inboxes[index], self._outbox, factory), daemon=True)
                self.processes.append(process)
                process.start()
            ready = {}
            deadline = time.monotonic() + self.timeout
            while len(ready) < len(self.devices):
                message = self._receive(deadline)
                if message[0] != "ready" or message[1] in ready:
                    raise RuntimeError("Unexpected encoder worker initialization message")
                ready[message[1]] = (message[2], message[3])
                print(f"Encoder worker ready on {self.devices[message[1]]}", flush=True)
            self.dimension = ready[0][0]
            if any(value[0] != self.dimension for value in ready.values()):
                raise ValueError("Encoder workers disagree on embedding dimension")
            self.worker_infos = [ready[i][1] for i in range(len(self.devices))]
            # Every replica must encode with identical weights and preprocessing.
            for key in ("model", "resolved_revision", "dimension", "max_tokens", "prompt",
                        "pooling_config_sha256", "padding_side", "dtype"):
                if any(info.get(key) != self.worker_infos[0].get(key) for info in self.worker_infos):
                    raise ValueError(f"Encoder workers disagree on {key}")
        except BaseException:
            self.close()
            raise

    @property
    def info(self):
        return {**self.worker_infos[0], "device": list(self.devices),
                "devices": list(self.devices), "strategy": "independent spawned encoder replicas",
                "per_device_batch_size": self.batch_size, "chunk_size": self.chunk_size,
                "dispatch_size": self.dispatch_size, "worker_timeout_seconds": self.timeout,
                "worker_review_counts": list(self.processed_counts)}

    def _collect_messages(self):
        # Queue.get's timeout does not cover a partially received pickle if a
        # worker dies mid-write. Keep that blocking read off the controller so
        # liveness checks and the deadline still work. The daemon cannot prevent
        # CLI shutdown after a broken transport; all workers are closed first.
        while not self._receiver_stop.is_set():
            try:
                message = self._outbox.get(timeout=.25)
            except Empty:
                continue
            except (EOFError, OSError, ValueError) as error:
                self._messages.put(("transport_error", str(error)))
                return
            self._messages.put(message)

    def _receive(self, deadline):
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Encoder worker timed out. Check GPU memory/load; reduce the "
                                   "encoding batch or chunk size, or increase --worker-timeout.")
            try:
                message = self._messages.get(timeout=min(.25, remaining))
            except Empty:
                dead = [(self.devices[i], p.exitcode) for i, p in enumerate(self.processes)
                        if p.exitcode is not None]
                if dead:
                    raise RuntimeError(f"Encoder worker exited unexpectedly: {dead}. "
                                       "Check GPU/CPU memory and worker logs.")
                continue
            if message[0] == "error":
                raise RuntimeError(f"Encoder worker on {self.devices[message[1]]} failed:\n{message[2]}")
            if message[0] == "transport_error":
                raise RuntimeError(f"Encoder worker communication failed: {message[1]}")
            return message

    def encode(self, texts):
        if self.closed:
            raise RuntimeError("Encoder pool is closed")
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        results = []
        try:
            for offset in range(0, len(texts), self.dispatch_size):
                window = texts[offset:offset + self.dispatch_size]
                jobs = {}
                for worker, start in enumerate(range(0, len(window), self.chunk_size)):
                    chunk = list(window[start:start + self.chunk_size])
                    job_id = self._job_id
                    self._job_id += 1
                    jobs[job_id] = (worker, len(chunk))
                    self._inboxes[worker].put((job_id, chunk), timeout=self.timeout)
                completed = {}
                deadline = time.monotonic() + self.timeout
                while len(completed) < len(jobs):
                    message = self._receive(deadline)
                    if message[0] != "result":
                        raise RuntimeError("Unexpected encoder worker response")
                    _, worker, job_id, values = message
                    if job_id not in jobs or job_id in completed or jobs[job_id][0] != worker:
                        raise RuntimeError("Mismatched encoder worker result")
                    if values.shape != (jobs[job_id][1], self.dimension):
                        raise ValueError("Encoder worker result has an invalid shape")
                    completed[job_id] = values
                    self.processed_counts[worker] += len(values)
                results.extend(completed[job_id] for job_id in jobs)
            return np.concatenate(results, axis=0)
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.closed:
            return
        self.closed = True
        self._receiver_stop.set()
        for inbox in self._inboxes:
            try:
                inbox.put_nowait(None)
            except (Full, OSError, ValueError):
                pass
        deadline = time.monotonic() + 2
        for process in self.processes:
            if process.pid is not None:
                process.join(timeout=max(0, deadline - time.monotonic()))
        for process in self.processes:
            if process.pid is not None and process.is_alive():
                process.terminate()
        for process in self.processes:
            if process.pid is not None:
                process.join(timeout=3)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=3)
        # Never wait for feeder threads that may have been interrupted by a crash.
        for queue in [*self._inboxes, self._outbox]:
            queue.cancel_join_thread()
            queue.close()
        self._receiver.join(timeout=1)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
