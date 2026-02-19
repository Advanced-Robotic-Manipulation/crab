"""
Crab Robot Web Control Panel — NiceGUI entry point.

Run on Aibek, access from any browser at http://aibek:8080
"""

import asyncio
import logging
import signal
import sys

from nicegui import app, ui

from control_panel import create_control_panel
from dataset_viewer import create_dataset_viewer
from process_manager import HostState, ProcessManager, SessionState

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def setup_shutdown_hooks(pm: ProcessManager):
    """Register cleanup on app shutdown and OS signals."""

    async def _cleanup():
        logger.info("Shutdown hook triggered — cleaning up processes...")
        await pm.cleanup_all()

    app.on_shutdown(_cleanup)

    def _signal_handler(sig, frame):
        logger.info(f"Signal {sig} received, initiating shutdown...")
        # Schedule cleanup in the event loop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(pm.cleanup_all())
        sys.exit(0)

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)


@ui.page("/")
async def index():
    pm = ProcessManager()

    # ── Dark mode ─────────────────────────────────────────────────────
    ui.dark_mode(True)

    # ── Header ────────────────────────────────────────────────────────
    with ui.header().classes("items-center justify-between"):
        ui.label("Crab Robot Control").classes("text-xl font-bold")
        with ui.row().classes("items-center gap-3"):
            ui.label("Host:").classes("text-sm")
            header_host_badge = ui.badge("--", color="grey")
            ui.label("Session:").classes("text-sm")
            header_session_badge = ui.badge("--", color="grey")

    # ── Tabs ──────────────────────────────────────────────────────────
    with ui.tabs().classes("w-full") as tabs:
        control_tab = ui.tab("Control Panel", icon="gamepad")
        dataset_tab = ui.tab("Datasets", icon="folder_open")

    with ui.tab_panels(tabs, value=control_tab).classes("w-full max-w-5xl mx-auto"):
        with ui.tab_panel(control_tab):
            cp_host_badge, cp_session_badge = create_control_panel()

        with ui.tab_panel(dataset_tab):
            create_dataset_viewer()

    # ── Header badge updater ──────────────────────────────────────────
    def update_header():
        hs = pm.host_state
        header_host_badge.text = hs.value
        if hs == HostState.RUNNING:
            header_host_badge.props("color=positive")
        elif hs in (HostState.STARTING, HostState.STOPPING):
            header_host_badge.props("color=warning")
        elif hs == HostState.ERROR:
            header_host_badge.props("color=negative")
        else:
            header_host_badge.props("color=grey")

        ss = pm.session_state
        header_session_badge.text = ss.value
        if ss == SessionState.RUNNING:
            header_session_badge.props("color=positive")
        elif ss in (SessionState.STARTING, SessionState.STOPPING):
            header_session_badge.props("color=warning")
        else:
            header_session_badge.props("color=grey")

    ui.timer(1.0, update_header)


def main():
    pm = ProcessManager()
    setup_shutdown_hooks(pm)

    logger.info(f"Crab webapp starting — local IP: {pm.local_ip}")
    logger.info("Access at http://aibek:8082 or http://%s:8082", pm.local_ip)

    ui.run(
        host="0.0.0.0",
        port=8082,
        title="Crab Robot Control",
        dark=True,
        show=False,
        reload=False,
    )


if __name__ == "__main__":
    main()
