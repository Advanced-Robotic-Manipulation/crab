"""
Control Panel tab for Crab robot web app.

Provides host management, session control (teleop/recording), and log viewers.
"""

import asyncio

from nicegui import ui

from urllib.parse import quote

from process_manager import (
    HARDNESS,
    RERUN_GRPC_PORT,
    RERUN_WEB_PORT,
    TASKS,
    HostState,
    ProcessManager,
    SessionMode,
    SessionState,
)


def create_control_panel():
    """Build the Control Panel tab UI, wired to the singleton ProcessManager."""
    pm = ProcessManager()

    # ── Orin Host Control ─────────────────────────────────────────────
    with ui.card().classes("w-full"):
        ui.label("ORIN HOST CONTROL").classes("text-lg font-bold")
        with ui.row().classes("items-center gap-4 w-full"):
            host_status = ui.badge("STOPPED", color="grey").classes("text-sm")
            ui.space()
            host_start_btn = ui.button("Start", icon="play_arrow", on_click=lambda: _start_host())
            host_stop_btn = ui.button("Stop", icon="stop", on_click=lambda: _stop_host()).props("color=negative")

    # ── Session Config ────────────────────────────────────────────────
    with ui.card().classes("w-full"):
        ui.label("SESSION CONFIG").classes("text-lg font-bold")

        mode_select = ui.select(
            [m.value for m in SessionMode],
            value=SessionMode.TELEOPERATION.value,
            label="Mode",
        ).classes("w-60")

        # Recording options container
        with ui.column().classes("w-full gap-2").bind_visibility_from(
            mode_select, "value", backward=lambda v: v == SessionMode.RECORDING.value
        ) as recording_opts:
            ui.separator()
            ui.label("Recording Options").classes("text-sm text-grey-6")

            with ui.row().classes("gap-4 w-full"):
                task_select = ui.select(
                    {k: v for k, v in TASKS.items()},
                    value=0,
                    label="Task",
                ).classes("w-48")
                hardness_select = ui.select(
                    {k: v for k, v in HARDNESS.items()},
                    value=1,
                    label="Hardness",
                ).classes("w-36")

            with ui.row().classes("gap-4 items-end w-full"):
                dataset_input = ui.input(label="Dataset name", placeholder="auto-generated if empty").classes("w-64")
                auto_btn = ui.button("Auto-name", on_click=lambda: _auto_name())
                episodes_input = ui.number(label="Episodes", value=10, min=1, max=999, step=1).classes("w-28")

        ui.separator()

        with ui.row().classes("items-center gap-4 w-full"):
            session_start_btn = ui.button(
                "Start Session", icon="play_arrow",
                on_click=lambda: _start_session(),
            ).props("color=positive")
            session_stop_btn = ui.button(
                "Stop Session", icon="stop",
                on_click=lambda: _stop_session(),
            ).props("color=negative")
            session_status = ui.badge("IDLE", color="grey").classes("text-sm")

    # ── Recording Episode Control ────────────────────────────────────
    rec_card = ui.card().classes("w-full")
    rec_card.set_visibility(False)
    with rec_card:
        ui.label("EPISODE CONTROL").classes("text-lg font-bold")
        with ui.row().classes("items-center gap-4 w-full"):
            rec_toggle_btn = ui.button(
                "Record Episode", icon="fiber_manual_record",
                on_click=lambda: asyncio.create_task(pm.toggle_recording()),
            ).props("color=red")
            rec_saving_spinner = ui.spinner("dots", size="lg", color="orange")
            rec_saving_spinner.set_visibility(False)
            rec_save_btn = ui.button(
                "Save & Finish", icon="save",
                on_click=lambda: asyncio.create_task(pm.save_and_quit_recording()),
            ).props("color=warning")
            rec_discard_btn = ui.button(
                "Discard Episode", icon="delete_outline",
                on_click=lambda: asyncio.create_task(pm.discard_recording()),
            ).props("color=red-10 outline")
        with ui.row().classes("items-center gap-4 w-full"):
            rec_status_label = ui.label("Waiting...").classes("text-grey-6")
            rec_episode_label = ui.label("").classes("font-bold")
            rec_frames_label = ui.label("").classes("text-grey-6")

    # ── Rerun Live Viewer ────────────────────────────────────────────
    rerun_card = ui.card().classes("w-full")
    rerun_card.set_visibility(False)
    rerun_connect_url = quote(f"rerun+http://{pm.local_ip}:{RERUN_GRPC_PORT}/proxy", safe="")
    rerun_viewer_url = f"http://{pm.local_ip}:{RERUN_WEB_PORT}/?url={rerun_connect_url}"
    with rerun_card:
        with ui.row().classes("items-center gap-4"):
            ui.label("LIVE VIEWER (Rerun)").classes("text-lg font-bold")
            ui.link("Open in new tab", rerun_viewer_url, new_tab=True).classes("text-sm")
        rerun_iframe = ui.html(
            f'<iframe src="{rerun_viewer_url}" '
            f'style="width:100%; height:600px; border:none;"></iframe>'
        )

    # ── Host Log ──────────────────────────────────────────────────────
    with ui.card().classes("w-full"):
        with ui.row().classes("items-center"):
            ui.label("HOST LOG").classes("text-lg font-bold")
            ui.button(icon="delete_sweep", on_click=lambda: host_log.clear()).props("flat dense")
        host_log = ui.log(max_lines=200).classes("w-full h-48")

    # ── Session Log ───────────────────────────────────────────────────
    with ui.card().classes("w-full"):
        with ui.row().classes("items-center"):
            ui.label("SESSION LOG").classes("text-lg font-bold")
            ui.button(icon="delete_sweep", on_click=lambda: session_log.clear()).props("flat dense")
        session_log = ui.log(max_lines=200).classes("w-full h-48")

    # ── DANGER ZONE ──────────────────────────────────────────────────
    with ui.card().classes("w-full").style("border: 2px solid #c62828"):
        with ui.row().classes("items-center gap-4 w-full"):
            ui.icon("warning", color="red").classes("text-3xl")
            ui.label("DANGER ZONE").classes("text-lg font-bold text-red-8")
            ui.space()
            reboot_spinner = ui.spinner("dots", size="lg", color="red")
            reboot_spinner.set_visibility(False)
            reboot_status_label = ui.label("").classes("text-red-8 font-bold")
            reboot_status_label.set_visibility(False)
            reboot_btn = ui.button(
                "REBOOT CRAB", icon="restart_alt",
                on_click=lambda: reboot_dialog.open(),
            ).props('color=red-10 glossy').classes("text-white font-bold")

    # Reboot confirmation dialog
    with ui.dialog() as reboot_dialog, ui.card().classes("items-center"):
        ui.icon("dangerous", color="red").classes("text-6xl")
        ui.label("REBOOT ORIN?").classes("text-2xl font-bold text-red-8")
        ui.label(
            "This will kill ALL running processes and reboot the robot. "
            "It takes ~90 seconds to come back."
        ).classes("text-center text-grey-7")
        ui.separator()
        with ui.row().classes("gap-4 justify-center"):
            ui.button("Cancel", on_click=reboot_dialog.close).props("flat")
            ui.button(
                "YES, REBOOT", icon="restart_alt",
                on_click=lambda: (_confirm_reboot()),
            ).props("color=red-10").classes("text-white font-bold")

    _prev_rebooting = {"val": False}

    def _confirm_reboot():
        reboot_dialog.close()
        asyncio.create_task(pm.reboot_orin())

    def _update_reboot_ui():
        if pm.rebooting:
            reboot_btn.set_visibility(False)
            reboot_spinner.set_visibility(True)
            reboot_status_label.set_visibility(True)
            reboot_status_label.text = "Rebooting..."
            _prev_rebooting["val"] = True
        elif _prev_rebooting["val"]:
            # Just finished rebooting
            reboot_btn.set_visibility(True)
            reboot_spinner.set_visibility(False)
            reboot_status_label.text = "Back online!"
            reboot_status_label.style("color: #2e7d32")
            ui.notify("Orin is back online!", type="positive", position="top")
            _prev_rebooting["val"] = False

    # ── Periodic state update ─────────────────────────────────────────
    def update_ui():
        # Host status badge
        hs = pm.host_state
        host_status.text = hs.value
        if hs == HostState.RUNNING:
            host_status.props("color=positive")
        elif hs in (HostState.STARTING, HostState.STOPPING):
            host_status.props("color=warning")
        elif hs == HostState.ERROR:
            host_status.props("color=negative")
        else:
            host_status.props("color=grey")

        # Host buttons — allow Start from STARTING (retry) and Stop from STARTING (abort)
        host_start_btn.set_enabled(hs in (HostState.STOPPED, HostState.ERROR, HostState.STARTING))
        host_stop_btn.set_enabled(hs in (HostState.RUNNING, HostState.ERROR, HostState.STARTING))

        # Session status badge
        ss = pm.session_state
        session_status.text = ss.value
        if ss == SessionState.RUNNING:
            session_status.props("color=positive")
        elif ss in (SessionState.STARTING, SessionState.STOPPING):
            session_status.props("color=warning")
        else:
            session_status.props("color=grey")

        # Check if a recording save is in progress (video encoding — must not be interrupted)
        _rstate = pm.get_record_state()
        _saving = _rstate.get('saving', False) if _rstate else False

        # Session buttons — disable Stop while save_episode() is encoding videos
        session_start_btn.set_enabled(ss == SessionState.IDLE and hs == HostState.RUNNING)
        session_stop_btn.set_enabled(
            ss in (SessionState.RUNNING, SessionState.STARTING) and not _saving
        )

        # Show Rerun iframe only for teleop (recording uses rr.spawn() — native only)
        rerun_card.set_visibility(
            ss == SessionState.RUNNING and pm.session_mode == SessionMode.TELEOPERATION
        )

        # Recording episode control
        is_recording_session = (
            ss == SessionState.RUNNING
            and pm.session_mode == SessionMode.RECORDING
        )
        rec_card.set_visibility(is_recording_session)
        if is_recording_session:
            rstate = _rstate  # reuse from save guard above
            if rstate:
                saving = rstate.get("saving", False)
                if saving:
                    rec_status_label.text = "SAVING..."
                    rec_status_label.classes(replace="text-orange font-bold")
                    rec_toggle_btn.set_visibility(False)
                    rec_saving_spinner.set_visibility(True)
                    rec_save_btn.set_enabled(False)
                    rec_discard_btn.set_enabled(False)
                elif rstate["recording"]:
                    rec_status_label.text = "RECORDING"
                    rec_status_label.classes(replace="text-red font-bold")
                    rec_toggle_btn.set_visibility(True)
                    rec_saving_spinner.set_visibility(False)
                    rec_toggle_btn.text = "Stop Episode"
                    rec_toggle_btn._props["icon"] = "stop"
                    rec_toggle_btn.props("color=grey")
                    rec_toggle_btn.set_enabled(True)
                    rec_save_btn.set_enabled(True)
                    rec_discard_btn.set_enabled(True)
                else:
                    rec_status_label.text = "Ready"
                    rec_status_label.classes(replace="text-grey-6")
                    rec_toggle_btn.set_visibility(True)
                    rec_saving_spinner.set_visibility(False)
                    rec_toggle_btn.text = "Record Episode"
                    rec_toggle_btn._props["icon"] = "fiber_manual_record"
                    rec_toggle_btn.props("color=red")
                    rec_toggle_btn.set_enabled(True)
                    rec_save_btn.set_enabled(True)
                    rec_discard_btn.set_enabled(False)
                rec_episode_label.text = f"Episode {rstate['episode']}/{rstate['total_episodes']}"
                frames = rstate["frames"]
                rec_frames_label.text = f"{frames} frames ({frames/30:.1f}s)" if frames else ""
                rec_toggle_btn.update()
            else:
                rec_status_label.text = "Starting..."
                rec_status_label.classes(replace="text-grey-6")
                rec_episode_label.text = ""
                rec_frames_label.text = ""

        # Drain buffered log messages into UI
        for msg in pm.drain_host_log():
            host_log.push(msg)
        for msg in pm.drain_session_log():
            session_log.push(msg)

        # Reboot spinner/status
        _update_reboot_ui()

    ui.timer(0.5, update_ui)

    # ── Action handlers (fire-and-forget so UI stays responsive) ─────
    def _start_host():
        asyncio.create_task(pm.start_host())

    def _stop_host():
        async def _do():
            if pm.session_state != SessionState.IDLE:
                await pm.stop_session()
            await pm.stop_host()
        asyncio.create_task(_do())

    def _start_session():
        mode = SessionMode(mode_select.value)
        asyncio.create_task(pm.start_session(
            mode=mode,
            task_id=int(task_select.value) if mode == SessionMode.RECORDING else 0,
            hardness_id=int(hardness_select.value) if mode == SessionMode.RECORDING else 1,
            dataset_name=dataset_input.value.strip() if mode == SessionMode.RECORDING else "",
            num_episodes=int(episodes_input.value) if mode == SessionMode.RECORDING else 10,
        ))

    def _stop_session():
        asyncio.create_task(pm.stop_session())

    def _auto_name():
        task_name = TASKS.get(int(task_select.value), "unknown")
        hardness_name = HARDNESS.get(int(hardness_select.value), "medium")
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        dataset_input.value = f"{task_name}_{hardness_name}_{ts}"

    return host_status, session_status
