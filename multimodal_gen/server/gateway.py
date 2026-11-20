"""
Dual-Protocol Gateway (OSC + JSON-RPC)

Starts both the OSC server (JUCE) and JSON-RPC server (Copilot-Liku CLI)
using a shared ServerConfig and a single shared GenerationWorker.
Callbacks are routed back to the originating protocol so JSON-RPC tasks
do not leak unsolicited OSC notifications into JUCE (and vice versa).
"""

from __future__ import annotations

import signal
import threading
import time
from pathlib import Path
from typing import Optional

from .config import ServerConfig
from .osc_server import MusicGenOSCServer
from .jsonrpc_server import MusicGenJSONRPCServer
from .worker import GenerationResult, GenerationWorker


class GatewayServer:
    """Boots OSC and JSON-RPC servers together on a shared GenerationWorker."""

    def __init__(
        self,
        osc_host: str = "127.0.0.1",
        osc_recv_port: int = 9000,
        osc_send_port: int = 9001,
        jsonrpc_host: str = "127.0.0.1",
        jsonrpc_port: int = 8765,
        verbose: bool = False,
        log_file: Optional[str] = None,
        config: Optional[ServerConfig] = None,
    ) -> None:
        self.config = config or ServerConfig(
            host=osc_host,
            recv_port=osc_recv_port,
            send_port=osc_send_port,
            verbose=verbose,
            log_file=log_file,
        )

        # Default log file for GUI-embedded runs (stdout isn't visible).
        if not self.config.log_file:
            try:
                base_output = self.config.default_output_dir or "output"
                self.config.log_file = str(Path(base_output) / "gateway.log")
            except Exception:
                pass

        # Keep gateway host/port attributes synchronized with self.config.
        self.osc_host = self.config.host
        self.osc_recv_port = self.config.recv_port
        self.osc_send_port = self.config.send_port
        self.jsonrpc_host = jsonrpc_host
        self.jsonrpc_port = jsonrpc_port

        self._osc_server: Optional[MusicGenOSCServer] = None
        self._json_server: Optional[MusicGenJSONRPCServer] = None
        self._worker: Optional[GenerationWorker] = None
        self._running = False

    # ------------------------------------------------------------------ #
    # Task-origin-aware callback routing
    # ------------------------------------------------------------------ #

    def _task_owner(self, task_id: Optional[str]) -> Optional[str]:
        """Identify which protocol server submitted *task_id* ('osc', 'jsonrpc', or None)."""
        if not task_id:
            return None

        in_osc = False
        if self._osc_server is not None:
            owned = getattr(self._osc_server, "_owned_task_ids", None)
            if isinstance(owned, set) and task_id in owned:
                in_osc = True

        in_json = False
        if self._json_server is not None:
            progress_lock = getattr(self._json_server, "_progress_lock", None)
            progress_map = getattr(self._json_server, "_progress", None)
            if progress_lock is not None and isinstance(progress_map, dict):
                with progress_lock:
                    in_json = task_id in progress_map
            elif isinstance(progress_map, dict):
                in_json = task_id in progress_map

        if in_osc and not in_json:
            return "osc"
        if in_json and not in_osc:
            return "jsonrpc"
        return None

    def _current_task_id(self) -> Optional[str]:
        if self._worker is None:
            return None
        return self._worker.get_current_task_id()

    def _dispatch_progress(self, step: str, percent: float, message: str) -> None:
        """Route progress updates to the originating protocol server."""
        owner = self._task_owner(self._current_task_id())
        if self._json_server and owner in ("jsonrpc", None):
            try:
                self._json_server._on_progress(step, percent, message)
            except Exception:
                pass
        if self._osc_server and owner in ("osc", None):
            try:
                self._osc_server._on_progress(step, percent, message)
            except Exception:
                pass

    def _dispatch_complete(self, result: GenerationResult) -> None:
        """Route completion to the originating protocol server."""
        task_id = getattr(result, "task_id", None)
        owner = self._task_owner(task_id)
        if self._json_server and owner in ("jsonrpc", None):
            try:
                self._json_server._on_complete(result)
            except Exception:
                pass
        if self._osc_server and owner in ("osc", None):
            try:
                self._osc_server._on_generation_complete(result)
            except Exception:
                pass
            if task_id:
                owned = getattr(self._osc_server, "_owned_task_ids", None)
                if isinstance(owned, set):
                    owned.discard(task_id)

    def _dispatch_error(self, code: int, message: str) -> None:
        """Route worker errors to the originating protocol server."""
        task_id = self._current_task_id()
        owner = self._task_owner(task_id)
        if self._json_server and owner in ("jsonrpc", None):
            try:
                self._json_server._on_error(code, message)
            except Exception:
                pass
        if self._osc_server and owner in ("osc", None):
            try:
                self._osc_server._on_error(code, message)
            except Exception:
                pass
            if task_id:
                owned = getattr(self._osc_server, "_owned_task_ids", None)
                if isinstance(owned, set):
                    owned.discard(task_id)

    def start(self) -> None:
        """Start both servers (blocking)."""
        self._start_servers()
        self._install_signal_handlers()
        self._running = True
        try:
            while self._running:
                time.sleep(0.5)
        except KeyboardInterrupt:
            self.stop()

    def start_async(self) -> None:
        """Start both servers in background threads."""
        self._start_servers(async_mode=True)
        self._running = True

    def stop(self) -> None:
        """Stop both servers and shut down the shared worker exactly once."""
        self._running = False
        shared_worker = self._worker
        try:
            if self._osc_server:
                # Detach shared worker so each server does not shut it down twice.
                if self._osc_server._gen_worker is shared_worker:
                    self._osc_server._gen_worker = None
                self._osc_server.stop()
        finally:
            try:
                if self._json_server:
                    if shared_worker is not None and self._json_server._worker is shared_worker:
                        # Prevent duplicate shutdown inside json_server.stop().
                        pass
                    self._json_server.stop()
            finally:
                if shared_worker is not None:
                    try:
                        shared_worker.shutdown(wait=True, timeout=10.0)
                    except Exception:
                        pass

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _start_servers(self, async_mode: bool = False) -> None:
        """Create and start OSC + JSON-RPC servers."""
        if self._worker is None:
            self._worker = GenerationWorker(
                max_workers=self.config.max_workers,
                progress_callback=self._dispatch_progress,
                completion_callback=self._dispatch_complete,
                error_callback=self._dispatch_error,
            )

        # OSC server for JUCE
        self._osc_server = MusicGenOSCServer(config=self.config, worker=self._worker)
        if async_mode:
            self._osc_server.start_async()
        else:
            threading.Thread(
                target=self._osc_server.start,
                kwargs={"handle_signals": False},
                name="Gateway-OSC",
                daemon=True,
            ).start()

        # JSON-RPC server for Copilot-Liku CLI
        self._json_server = MusicGenJSONRPCServer(
            config=self.config,
            host=self.jsonrpc_host,
            port=self.jsonrpc_port,
            worker=self._worker,
        )
        if async_mode:
            self._json_server.start_async()
        else:
            threading.Thread(
                target=self._json_server.start,
                name="Gateway-JSONRPC",
                daemon=True,
            ).start()

        print(
            f"[Gateway] OSC {self.osc_host}:{self.osc_recv_port}/{self.osc_send_port} "
            f"+ JSON-RPC {self.jsonrpc_host}:{self.jsonrpc_port} started"
        )

    def _install_signal_handlers(self) -> None:
        def _handler(signum: int, _frame: object) -> None:
            print(f"[Gateway] Caught signal {signum}, stopping...")
            self.stop()

        try:
            signal.signal(signal.SIGINT, _handler)
            signal.signal(signal.SIGTERM, _handler)
        except Exception:
            # Some environments (Windows) may not support all signals
            pass


def run_gateway(
    osc_host: str = "127.0.0.1",
    osc_recv_port: int = 9000,
    osc_send_port: int = 9001,
    jsonrpc_host: str = "127.0.0.1",
    jsonrpc_port: int = 8765,
    verbose: bool = False,
    log_file: Optional[str] = None,
) -> None:
    """Convenience entry to run the dual-protocol gateway (blocking)."""
    gw = GatewayServer(
        osc_host=osc_host,
        osc_recv_port=osc_recv_port,
        osc_send_port=osc_send_port,
        jsonrpc_host=jsonrpc_host,
        jsonrpc_port=jsonrpc_port,
        verbose=verbose,
        log_file=log_file,
    )
    gw.start()
