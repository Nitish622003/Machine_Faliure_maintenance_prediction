
"""
Live Smart Manufacturing Digital Twin Dashboard.

This is the application layer for the project. It executes the validated
local runtime engine event-by-event and then enriches actionable events
with machine memory, RAG evidence, and the TCS MaaS LLM.

Run from the repository root:

    streamlit run app/live_dashboard.py

The deterministic Smart Twin + orchestrator result remains authoritative.
RAG/LLM provide supporting knowledge and explanation.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import numpy as np
import streamlit as st


# ============================================================
# PROJECT PATH
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.realtime_engine import RuntimeEngine
from src.rag_engine import MaintenanceRAG
from src.llm_engine import MaintenanceLLM


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Smart Manufacturing Digital Twin",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# SESSION STATE
# ============================================================

DEFAULTS = {
    "engine": None,
    "rag": None,
    "llm": None,
    "running": False,
    "last_event": None,
    "event_log": [],
    "llm_report": None,
    "last_retrieved": [],
    "rag_retrieval_history": [],
    "chat_history": [],
    "auto_llm": True,
    "last_actionable_event": None,
    "last_alert_popup_key": None,
    "test_alert_active": False,
    "chat_open": False,
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# INITIALIZATION
# ============================================================

def initialize_engines():
    if st.session_state.engine is None:
        st.session_state.engine = RuntimeEngine.create(ROOT)

    if st.session_state.rag is None:
        st.session_state.rag = MaintenanceRAG.from_project(
            ROOT,
            top_k=3,
        )

    if st.session_state.llm is None:
        try:
            st.session_state.llm = MaintenanceLLM()
        except Exception:
            # API key may be absent until the user launches the terminal
            # with the environment variable. The rest of the UI can still
            # execute the deterministic pipeline.
            st.session_state.llm = None


try:
    initialize_engines()
except Exception as exc:
    st.error(f"Application initialization failed: {exc}")
    st.stop()


engine: RuntimeEngine = st.session_state.engine
rag: MaintenanceRAG = st.session_state.rag


# ============================================================
# HELPERS
# ============================================================

def status_icon(status: Any) -> str:
    status = str(status).upper()
    return {
        "HEALTHY": "🟢",
        "WARNING": "🟠",
        "CRITICAL": "🔴",
    }.get(status, "⚪")


def format_value(value: Any, digits: int = 2) -> str:
    if value is None:
        return "—"

    try:
        if pd.isna(value):
            return "—"
    except Exception:
        pass

    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def get_machine_memory_context(machine_id: Any, max_events: int = 5) -> str:
    """Return recent runtime history for one machine for LLM context."""
    machine_id = str(machine_id) if machine_id is not None else ""
    events = [
        e for e in st.session_state.get("event_log", [])
        if str(e.get("Machine_ID", "")) == machine_id
    ]

    if not events:
        return "No previous runtime events are available for this machine."

    recent = events[-max_events:]
    lines = []
    for e in recent:
        lines.append(
            f"Machine={e.get('Machine_ID', 'UNKNOWN')} | "
            f"Cycle={e.get('Cycle', 'UNKNOWN')} | "
            f"Fault={e.get('Simulation_Fault', 'NONE')} | "
            f"Risk={e.get('failure_risk', 'UNKNOWN')} | "
            f"RUL={e.get('predicted_rul_cycles', 'UNKNOWN')} | "
            f"Status={e.get('overall_status', 'UNKNOWN')} | "
            f"Decision={e.get('maintenance_decision', 'NO ACTION')} | "
            f"Alert={e.get('alert_level', 'NONE')}"
        )

    return "\n".join(lines)


def get_alert_history_context(max_alerts: int = 10, max_degradations: int = 10) -> str:
    """Build plant-wide alert/degradation context for the Maintenance Assistant."""
    history = pd.DataFrame(st.session_state.get("event_log", []))

    if history.empty:
        return "No runtime alert or degradation history is available yet."

    lines = []

    # Alert history: only real WARNING/CRITICAL alerts.
    if {"alert_level", "alert_required"}.issubset(history.columns):
        alerts = history[
            history["alert_level"].astype(str).str.upper().isin(["WARNING", "CRITICAL"])
            & history["alert_required"].fillna(False).astype(bool)
        ].copy()
    else:
        alerts = pd.DataFrame()

    if not alerts.empty:
        lines.append(f"Total generated alerts in current session: {len(alerts)}")
        critical = int(
            (alerts["alert_level"].astype(str).str.upper() == "CRITICAL").sum()
        )
        warning = int(
            (alerts["alert_level"].astype(str).str.upper() == "WARNING").sum()
        )
        lines.append(f"Critical alerts: {critical}")
        lines.append(f"Warning alerts: {warning}")
        lines.append("Recent generated alerts:")
        for _, row in alerts.tail(max_alerts).iloc[::-1].iterrows():
            lines.append(
                f"- {row.get('Machine_ID', 'UNKNOWN')} | "
                f"Cycle {row.get('Cycle', 'UNKNOWN')} | "
                f"Fault={row.get('Simulation_Fault', 'NONE')} | "
                f"Level={row.get('alert_level', 'NONE')} | "
                f"Decision={row.get('maintenance_decision', 'NO ACTION')} | "
                f"Action={row.get('maintenance_action', 'UNKNOWN')} | "
                f"Message={row.get('alert_message', '')}"
            )
    else:
        lines.append("Total generated alerts in current session: 0")

    # Degradation history: WARNING/CRITICAL, MEDIUM/HIGH risk, or RUL <= 50.
    detected_mask = pd.Series(False, index=history.index)
    if "overall_status" in history.columns:
        detected_mask |= history["overall_status"].astype(str).str.upper().isin(
            ["WARNING", "CRITICAL"]
        )
    if "failure_risk" in history.columns:
        detected_mask |= history["failure_risk"].astype(str).str.upper().isin(
            ["MEDIUM", "HIGH"]
        )
    if "predicted_rul_cycles" in history.columns:
        detected_mask |= (
            pd.to_numeric(history["predicted_rul_cycles"], errors="coerce") <= 50
        )

    detected = history[detected_mask].copy()

    lines.append(f"Detected degradation/actionable-condition events: {len(detected)}")
    if not detected.empty:
        lines.append("Recent degradation/actionable events:")
        for _, row in detected.tail(max_degradations).iloc[::-1].iterrows():
            lines.append(
                f"- {row.get('Machine_ID', 'UNKNOWN')} | "
                f"Cycle {row.get('Cycle', 'UNKNOWN')} | "
                f"Fault={row.get('Simulation_Fault', 'NONE')} | "
                f"Risk={row.get('failure_risk', 'UNKNOWN')} | "
                f"Status={row.get('overall_status', 'UNKNOWN')} | "
                f"RUL={row.get('predicted_rul_cycles', 'UNKNOWN')} | "
                f"Decision={row.get('maintenance_decision', 'NO ACTION')}"
            )

    return "\n".join(lines)


def ingest_runtime_event_into_rag(event: Dict[str, Any]) -> None:
    """Add the authoritative live event to the in-memory RAG corpus.

    The static rag_maintenance_reports.csv is never modified. Runtime
    machine evidence exists only for the current Streamlit session.
    """
    try:
        rag.add_runtime_event(event)
    except Exception as exc:
        # RAG must never stop the deterministic runtime pipeline.
        st.session_state["rag_ingest_error"] = str(exc)


def run_llm_for_event(event: Dict[str, Any]):
    """
    Retrieve live/static RAG evidence first, then optionally ask the LLM
    to explain it. RAG retrieval must remain available even when the
    MaaS LLM/API is unavailable.
    """
    # Build machine context for retrieval from the authoritative event.
    machine_context = {
        "Machine_ID": event.get("Machine_ID"),
        "Machine_Type": event.get("Machine_Type"),
        "Cycle": event.get("Cycle"),
        "Simulation_Fault": event.get("Simulation_Fault"),
        "failure_risk": event.get("failure_risk"),
        "overall_status": event.get("overall_status"),
        "predicted_rul_cycles": event.get("predicted_rul_cycles"),
        "failure_probability": event.get("failure_probability"),
        "maintenance_decision": event.get("maintenance_decision"),
        "alert_level": event.get("alert_level"),
    }

    # IMPORTANT: RAG retrieval is independent from LLM availability.
    try:
        retrieved = rag.retrieve(
            query=(
                f"{event.get('Simulation_Fault', '')} "
                f"{event.get('overall_status', '')} "
                f"{event.get('maintenance_decision', '')} "
                "maintenance machine history recommended action"
            ),
            machine_context=machine_context,
            top_k=3,
        )
    except Exception as exc:
        retrieved = []
        st.session_state["rag_retrieval_error"] = str(exc)

    # Preserve the retrieved evidence even if the LLM cannot be called.
    st.session_state.last_retrieved = retrieved

    # Build runtime memory context separately.
    memory_context = get_machine_memory_context(event.get("Machine_ID"))

    # If the LLM/API is unavailable, return a useful report while retaining
    # the RAG evidence in the UI.
    if st.session_state.llm is None:
        return (
            "LLM unavailable. RAG retrieval was completed and the retrieved "
            "maintenance evidence is shown in the RAG Evidence panel.",
            retrieved,
        )

    try:
        report = st.session_state.llm.generate_maintenance_report(
            machine_context=event,
            orchestrator_result={
                key: event.get(key)
                for key in [
                    "agents_invoked",
                    "trigger_reason",
                    "maintenance_priority",
                    "maintenance_decision",
                    "maintenance_reason",
                    "alert_level",
                    "alert_required",
                    "alert_message",
                    "maintenance_action",
                ]
            },
            memory_context=memory_context,
            rag_context=rag.format_context(retrieved),
        )
        return report, retrieved
    except Exception as exc:
        return (
            f"LLM generation failed: {exc}\n\n"
            "RAG retrieval was completed and remains available in the "
            "RAG Evidence panel.",
            retrieved,
        )

def process_until_next_real_alert(max_events: int = 1000):
    """Advance the real RuntimeEngine until the next real alert."""
    processed = 0

    while processed < max_events:
        event = engine.next_event()

        if event is None:
            st.session_state.running = False
            return None, processed

        st.session_state.last_event = event
        st.session_state.event_log.append(event)
        ingest_runtime_event_into_rag(event)
        processed += 1

        level = str(event.get("alert_level", "NONE")).upper()
        required = bool(event.get("alert_required", False))

        if level in {"WARNING", "CRITICAL"} and required:
            st.session_state.last_actionable_event = event.copy()

            if bool(event.get("agents_invoked", False)):
                report, retrieved = run_llm_for_event(event)
                st.session_state.llm_report = report
                st.session_state.last_retrieved = retrieved
                st.session_state.rag_retrieval_history.append(
                    {"event": event.copy(), "retrieved": retrieved}
                )

            return event, processed

    return None, processed


def process_one_event():
    """Execute exactly one real runtime event."""
    event = engine.next_event()

    if event is None:
        st.session_state.running = False
        return None

    st.session_state.last_event = event

    # Store the authoritative runtime result locally for the session.
    st.session_state.event_log.append(event)

    # Bridge the live simulation into RAG. This creates searchable
    # LIVE_RUNTIME_EVENT evidence without changing the static CSV.
    ingest_runtime_event_into_rag(event)

    # Only enrich actionable events. This prevents unnecessary LLM calls.
    actionable = bool(event.get("agents_invoked", False))

    if actionable:
        report, retrieved = run_llm_for_event(event)
        st.session_state.llm_report = report
        st.session_state.last_retrieved = retrieved
        st.session_state.last_actionable_event = event.copy()
        st.session_state.rag_retrieval_history.append(
            {"event": event.copy(), "retrieved": retrieved}
        )

    # Keep the latest actionable intelligence visible when the next
    # interleaved event from another machine is healthy.
    return event


def reset_application():
    engine.reset()
    try:
        rag.clear_runtime_events()
    except Exception:
        pass

    st.session_state.running = False
    st.session_state.last_event = None
    st.session_state.event_log = []
    st.session_state.llm_report = None
    st.session_state.last_retrieved = []
    st.session_state.rag_retrieval_history = []
    st.session_state.chat_history = []
    st.session_state.last_actionable_event = None
    st.session_state.last_alert_popup_key = None
    st.session_state.test_alert_active = False
    st.session_state.chat_open = False


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🎛️ Simulation Control")

st.sidebar.metric(
    "Processed events",
    f"{engine.event_index} / {engine.total_events}",
)

speed = st.sidebar.select_slider(
    "Playback delay",
    options=[
        0.25,
        0.5,
        1.0,
        2.0,
        3.0,
    ],
    value=1.0,
    format_func=lambda x: f"{x:g}s",
)

st.session_state.auto_llm = st.sidebar.checkbox(
    "Generate LLM report on actionable events",
    value=st.session_state.auto_llm,
)

if st.sidebar.button(
    "▶ Start live simulation",
    use_container_width=True,
):
    st.session_state.running = True
    st.rerun()

if st.sidebar.button(
    "⏭ Process one event",
    use_container_width=True,
):
    process_one_event()
    st.rerun()


if st.sidebar.button(
    "🚨 Jump to next real alert",
    use_container_width=True,
    help="Fast-forward the real runtime until the next WARNING/CRITICAL alert.",
):
    found_alert, skipped = process_until_next_real_alert()
    if found_alert is None:
        st.sidebar.warning(
            "No further real alert exists in the remaining simulation."
        )
    else:
        level = str(found_alert.get("alert_level", "WARNING")).upper()
        st.sidebar.success(
            f"Found real {level} alert after {skipped} events."
        )
    st.rerun()


if st.sidebar.button(
    "🧪 Test alert popup",
    use_container_width=True,
    help="UI-only test. Does not modify Smart Twin or runtime data.",
):
    st.session_state.test_alert_active = True
    st.rerun()


if st.sidebar.button(
    "⏸ Pause",
    use_container_width=True,
):
    st.session_state.running = False
    st.rerun()

if st.sidebar.button(
    "↺ Reset",
    use_container_width=True,
):
    reset_application()
    st.rerun()


# ============================================================
# SIDEBAR ALERT CENTER
# ============================================================

with st.sidebar:
    st.markdown("### 🚨 Alert Center")

    sidebar_history = pd.DataFrame(st.session_state.event_log)

    if not sidebar_history.empty and "alert_level" in sidebar_history.columns:
        sidebar_alerts = sidebar_history[
            sidebar_history["alert_level"].astype(str).str.upper().isin(
                ["WARNING", "CRITICAL"]
            )
        ].tail(8).iloc[::-1]

        if not sidebar_alerts.empty:
            for _, row in sidebar_alerts.iterrows():
                level = str(row.get("alert_level", "WARNING")).upper()
                icon = "🚨" if level == "CRITICAL" else "⚠️"
                st.markdown(
                    f"{icon} **{level}**  \n"
                    f"`{row.get('Machine_ID', '—')}` · "
                    f"Cycle `{row.get('Cycle', '—')}`  \n"
                    f"{row.get('maintenance_decision', '—')}"
                )
                st.divider()
        else:
            st.success("🟢 No alerts recorded yet.")
    else:
        st.info("Alerts will appear here during simulation.")


# ============================================================
# HEADER
# ============================================================

event = st.session_state.last_event

if event:
    machine = event.get("Machine_ID", "—")
    cycle = event.get("Cycle", "—")
    overall_status = event.get(
        "overall_status",
        "UNKNOWN",
    )
else:
    machine = "—"
    cycle = "—"
    overall_status = "READY"

st.title("🏭 Smart Manufacturing Digital Twin")
st.markdown(
    f"## {status_icon(overall_status)} "
    f"{machine} · Cycle {cycle}"
)

st.caption(
    "LIVE LOCAL PIPELINE · IoT → Smart Twin → Agents → "
    "Orchestrator → Memory → RAG → MaaS LLM"
)


# ============================================================
# EXECUTION STATUS
# ============================================================

st.divider()
st.subheader("⚡ Runtime Execution")

stages = [
    ("📡", "IoT Event"),
    ("🧠", "Smart Twin"),
    ("🤖", "Agents"),
    ("🎯", "Orchestrator"),
    ("🧠", "Memory"),
    ("📚", "RAG"),
    ("💬", "MaaS LLM"),
]

cols = st.columns(len(stages))

for i, (emoji, label) in enumerate(stages):
    with cols[i]:
        if event is None:
            st.info(f"{emoji} **{label}**\n\nWaiting")
        elif label in {"Memory", "RAG", "MaaS LLM"}:
            if event.get("agents_invoked"):
                st.success(f"{emoji} **{label}**\n\nProcessed")
            else:
                st.info(f"{emoji} **{label}**\n\nNot required")
        else:
            st.success(f"{emoji} **{label}**\n\nProcessed")


# ============================================================
# CURRENT EVENT
# ============================================================

if event is None:
    st.info(
        "Press **Process one event** to execute the first IoT event, "
        "or press **Start live simulation**."
    )
else:
    st.divider()
    st.subheader("📡 Live IoT Telemetry")

    telemetry = [
        ("Temperature", "Temperature_C", "°C"),
        ("Vibration", "Vibration_mm_s", "mm/s"),
        ("Pressure", "Pressure_bar", "bar"),
        ("Rotation", "Rotational_Speed_RPM", "RPM"),
        ("Load", "Load_Fraction", ""),
        ("Torque", "Torque_Nm", "Nm"),
        ("Current", "Current_A", "A"),
        ("Acoustic", "Acoustic_dB", "dB"),
    ]

    cols = st.columns(4)

    for i, (label, key, unit) in enumerate(telemetry):
        with cols[i % 4]:
            value = event.get(key)
            st.metric(
                label,
                f"{format_value(value)} {unit}".strip(),
            )

    st.write(
        f"**Machine:** {event.get('Machine_ID')}  |  "
        f"**Type:** {event.get('Machine_Type')}  |  "
        f"**Cycle:** {event.get('Cycle')}  |  "
        f"**Simulation fault:** {event.get('Simulation_Fault')}"
    )


# ============================================================
# SMART TWIN
# ============================================================

if event:
    st.divider()
    st.subheader("🧠 Smart Twin Prediction")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Failure Probability",
            format_value(
                event.get("failure_probability"),
                4,
            ),
        )

    with c2:
        st.metric(
            "Failure Risk",
            str(event.get("failure_risk", "—")),
        )

    with c3:
        st.metric(
            "Predicted RUL",
            f"{format_value(event.get('predicted_rul_cycles'))} cycles",
        )

    with c4:
        st.metric(
            "Health Score",
            format_value(event.get("health_score")),
        )

    p1, p2, p3 = st.columns(3)

    with p1:
        st.write(
            f"**Failure prediction:** "
            f"{event.get('failure_prediction')}"
        )

    with p2:
        st.write(
            f"**RUL status:** "
            f"{event.get('rul_status')}"
        )

    with p3:
        st.write(
            f"**Overall status:** "
            f"{status_icon(event.get('overall_status'))} "
            f"{event.get('overall_status')}"
        )

    st.write(
        f"**Maintenance action:** "
        f"{event.get('maintenance_action')}"
    )


# ============================================================
# ALERTS + AGENT & ORCHESTRATOR
# ============================================================

if event:
    st.divider()
    st.subheader("🚨 Alerts & 🤖 Agent / Orchestrator Decision")

    current_level = str(event.get("alert_level", "NONE")).upper()
    current_status = str(event.get("overall_status", "HEALTHY")).upper()
    current_invoked = bool(event.get("agents_invoked", False))
    current_trigger = str(event.get("trigger_reason", "NORMAL"))
    current_decision = str(event.get("maintenance_decision", "NO ACTION"))
    current_action = str(event.get("maintenance_action", "NORMAL OPERATION"))

    if current_level in {"WARNING", "CRITICAL"} and bool(
        event.get("alert_required", False)
    ):
        alert_key = (
            f"{event.get('Machine_ID')}|{event.get('Cycle')}|"
            f"{current_level}|{current_decision}|"
            f"{event.get('alert_message')}"
        )

        if st.session_state.last_alert_popup_key != alert_key:
            st.toast(
                f"{'CRITICAL' if current_level == 'CRITICAL' else 'WARNING'} "
                f"— {event.get('Machine_ID')} · Cycle {event.get('Cycle')}",
                icon="🚨" if current_level == "CRITICAL" else "⚠️",
            )
            st.session_state.last_alert_popup_key = alert_key

        alert_machine = event.get("Machine_ID", "—")
        alert_cycle = event.get("Cycle", "—")
        alert_message = event.get("alert_message") or (
            "Immediate maintenance required."
            if current_level == "CRITICAL"
            else "Maintenance attention required."
        )
        alert_class = (
            "critical-alert-card" if current_level == "CRITICAL"
            else "warning-alert-card"
        )
        alert_icon = "🚨" if current_level == "CRITICAL" else "⚠️"
        alert_title = (
            "CRITICAL ALERT · ACTION REQUIRED"
            if current_level == "CRITICAL"
            else "MAINTENANCE WARNING · INSPECTION REQUIRED"
        )
        st.markdown(
            f"""<div class=\"{alert_class}\">
            <div class=\"alert-title\">{alert_icon} {alert_title}</div>
            <div class=\"alert-meta\">
            <b>Machine:</b> {alert_machine} &nbsp;·&nbsp; <b>Cycle:</b> {alert_cycle}<br>
            <b>Decision:</b> {current_decision}<br>
            <b>Action:</b> {current_action}<br>
            <b>Message:</b> {alert_message}
            </div></div>""",
            unsafe_allow_html=True,
        )

    if st.session_state.test_alert_active:
        st.warning(
            "🧪 **TEST ALERT — UI ONLY**  \n"
            "The popup/banner system is working. This is not a real machine alert."
        )
        if st.button("Clear test alert", key="clear_test_alert"):
            st.session_state.test_alert_active = False
            st.rerun()

    left, right = st.columns(2)

    with left:
        st.markdown("### Agent Layer")
        if current_invoked:
            st.success("Agents INVOKED")
        elif current_trigger == "ALREADY_ACTIONABLE":
            st.warning("Agents not reinvoked — condition is already actionable")
        else:
            st.info("Agents NOT INVOKED — current event is normal")

        st.write(f"**Trigger:** {current_trigger}")
        st.write(f"**Priority:** {event.get('maintenance_priority', 'NONE')}")
        st.write(f"**Decision:** {current_decision}")
        st.write(f"**Reason:** {event.get('maintenance_reason') or 'None'}")

    with right:
        st.markdown("### Alert Layer")
        if current_level == "CRITICAL":
            st.error("🔴 CRITICAL")
        elif current_level == "WARNING":
            st.warning("🟠 WARNING")
        else:
            st.success("🟢 NONE")

        st.write(f"**Required:** {event.get('alert_required', False)}")
        st.write(
            f"**Message:** {event.get('alert_message') or 'No new alert on this event'}"
        )

    # The simulation interleaves seven machines. Keep the latest active
    # maintenance condition visible even when the current event is healthy.
    history_df = pd.DataFrame(st.session_state.event_log)
    if not history_df.empty:
        actionable_mask = (
            history_df["overall_status"].astype(str).str.upper().isin(
                ["WARNING", "CRITICAL"]
            )
            | history_df["failure_risk"].astype(str).str.upper().isin(
                ["MEDIUM", "HIGH"]
            )
            | (
                pd.to_numeric(
                    history_df["predicted_rul_cycles"], errors="coerce"
                ) <= 50
            )
        )
        active = history_df[actionable_mask]

        if not active.empty:
            latest_active = active.iloc[-1]
            active_status = str(
                latest_active.get("overall_status", "WARNING")
            ).upper()

            banner = (
                f"`{latest_active.get('Machine_ID')}` · "
                f"Cycle `{latest_active.get('Cycle')}`  \n"
                f"Status: **{active_status}** · "
                f"Risk: **{latest_active.get('failure_risk')}** · "
                f"RUL: **{format_value(latest_active.get('predicted_rul_cycles'))} cycles**  \n"
                f"Decision: **{latest_active.get('maintenance_decision')}**"
            )

            latest_machine = latest_active.get("Machine_ID", "—")
            latest_cycle = latest_active.get("Cycle", "—")
            latest_risk = latest_active.get("failure_risk", "—")
            latest_rul = format_value(latest_active.get("predicted_rul_cycles"), 2)
            latest_decision = latest_active.get("maintenance_decision", "—")
            latest_message = latest_active.get("alert_message") or "Review the maintenance decision and monitor the machine."

            if active_status == "CRITICAL":
                st.markdown(
                    f"""<div class="operator-alert-card operator-alert-critical">
                    <div class="operator-alert-title">🚨 CRITICAL · ACTION REQUIRED</div>
                    <div class="operator-alert-meta"><b>{latest_machine}</b> · Cycle <b>{latest_cycle}</b><br>
                    Risk: <b>{latest_risk}</b> · RUL: <b>{latest_rul} cycles</b><br>
                    Decision: <b>{latest_decision}</b><br>
                    {latest_message}</div></div>""",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""<div class="operator-alert-card operator-alert-warning">
                    <div class="operator-alert-title">⚠️ WARNING · INSPECTION / MONITORING</div>
                    <div class="operator-alert-meta"><b>{latest_machine}</b> · Cycle <b>{latest_cycle}</b><br>
                    Risk: <b>{latest_risk}</b> · RUL: <b>{latest_rul} cycles</b><br>
                    Decision: <b>{latest_decision}</b><br>
                    {latest_message}</div></div>""",
                    unsafe_allow_html=True,
                )


# ============================================================
# ALERT & FAILURE DETECTION HISTORY
# ============================================================

if st.session_state.event_log:
    history = pd.DataFrame(st.session_state.event_log)

    detected_mask = (
        history["overall_status"].astype(str).str.upper().isin(
            ["WARNING", "CRITICAL"]
        )
        | history["failure_risk"].astype(str).str.upper().isin(
            ["MEDIUM", "HIGH"]
        )
        | (
            pd.to_numeric(
                history["predicted_rul_cycles"], errors="coerce"
            ) <= 50
        )
    )

    detected = history[detected_mask].copy()

    alerts = history[
        history["alert_level"].astype(str).str.upper().isin(
            ["WARNING", "CRITICAL"]
        )
        & history["alert_required"].fillna(False).astype(bool)
    ].copy()

    st.divider()
    st.subheader("🚨 Alert & Failure Detection History")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Alerts Generated", f"{len(alerts):,}")
    with c2:
        st.metric("Detected Degradation", f"{len(detected):,}")
    with c3:
        critical_count = (
            (
                alerts["alert_level"].astype(str).str.upper()
                == "CRITICAL"
            ).sum()
            if not alerts.empty else 0
        )
        st.metric("Critical Alerts", f"{critical_count:,}")
    with c4:
        warning_count = (
            (
                alerts["alert_level"].astype(str).str.upper()
                == "WARNING"
            ).sum()
            if not alerts.empty else 0
        )
        st.metric("Warning Alerts", f"{warning_count:,}")

    tab_alerts, tab_failures, tab_machines = st.tabs(
        ["🚨 All Alerts", "⚠️ Failures / Degradation", "🏭 Machine Summary"]
    )

    display_cols = [
        "Cycle",
        "Machine_ID",
        "Simulation_Fault",
        "failure_probability",
        "failure_risk",
        "predicted_rul_cycles",
        "overall_status",
        "agents_invoked",
        "trigger_reason",
        "maintenance_decision",
        "maintenance_action",
        "alert_level",
        "alert_required",
        "alert_message",
    ]

    with tab_alerts:
        if alerts.empty:
            st.info(
                "No real alerts have been generated in the processed events yet. "
                "Use '🚨 Jump to next real alert' to find the next real alert."
            )
        else:
            st.dataframe(
                alerts[
                    [c for c in display_cols if c in alerts.columns]
                ].sort_values("Cycle", ascending=False),
                use_container_width=True,
                hide_index=True,
            )

    with tab_failures:
        if detected.empty:
            st.info("No degradation/failure condition detected yet.")
        else:
            st.dataframe(
                detected[
                    [c for c in display_cols if c in detected.columns]
                ].sort_values("Cycle", ascending=False),
                use_container_width=True,
                hide_index=True,
            )

    with tab_machines:
        if detected.empty:
            st.info(
                "Machine-level statistics will appear as events are processed."
            )
        else:
            machine_summary = (
                detected.groupby("Machine_ID")
                .agg(
                    detected_events=("Machine_ID", "size"),
                    warning_events=(
                        "overall_status",
                        lambda s: (
                            s.astype(str).str.upper() == "WARNING"
                        ).sum(),
                    ),
                    critical_events=(
                        "overall_status",
                        lambda s: (
                            s.astype(str).str.upper() == "CRITICAL"
                        ).sum(),
                    ),
                    minimum_rul=("predicted_rul_cycles", "min"),
                    maximum_failure_probability=(
                        "failure_probability",
                        "max",
                    ),
                )
                .reset_index()
                .sort_values(
                    ["critical_events", "detected_events"],
                    ascending=False,
                )
            )

            st.dataframe(
                machine_summary,
                use_container_width=True,
                hide_index=True,
            )


# ============================================================
# MEMORY
# ============================================================

if event:
    st.divider()
    st.subheader("🧠 Runtime Machine Memory")

    machine_id = str(event["Machine_ID"])

    history = pd.DataFrame(
        st.session_state.event_log
    )

    if not history.empty:
        machine_history = history[
            history["Machine_ID"].astype(str) == machine_id
        ].tail(10)

        if len(machine_history) > 1:
            previous = machine_history.iloc[-2]

            m1, m2, m3, m4 = st.columns(4)

            with m1:
                st.metric(
                    "Previous Cycle",
                    str(previous.get("Cycle", "—")),
                )

            with m2:
                current_prob = float(
                    event.get(
                        "failure_probability",
                        0,
                    )
                )
                previous_prob = float(
                    previous.get(
                        "failure_probability",
                        0,
                    )
                )
                st.metric(
                    "Failure Probability Δ",
                    format_value(
                        current_prob - previous_prob,
                        4,
                    ),
                )

            with m3:
                current_rul = float(
                    event.get(
                        "predicted_rul_cycles",
                        0,
                    )
                )
                previous_rul = float(
                    previous.get(
                        "predicted_rul_cycles",
                        0,
                    )
                )
                st.metric(
                    "RUL Δ",
                    format_value(
                        current_rul - previous_rul,
                        2,
                    ),
                )

            with m4:
                changed = (
                    previous.get("overall_status")
                    != event.get("overall_status")
                )
                st.metric(
                    "Status Changed",
                    "YES" if changed else "NO",
                )

            st.dataframe(
                machine_history[
                    [
                        c
                        for c in [
                            "Cycle",
                            "failure_probability",
                            "predicted_rul_cycles",
                            "overall_status",
                            "maintenance_decision",
                        ]
                        if c in machine_history.columns
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info(
                "No previous event for this machine yet. "
                "Memory will become meaningful as the simulation advances."
            )



# ============================================================
# VISUALIZATION + FORECAST HELPERS
# ============================================================

@st.cache_data
def load_forecast_artifacts(root: Path):
    """Load the project's validated time-series forecast artifacts."""
    forecast_path = root / "data" / "processed" / "time_series_forecasts.csv"
    metrics_path = root / "data" / "processed" / "time_series_forecast_metrics.csv"

    forecast_df = pd.read_csv(forecast_path) if forecast_path.exists() else pd.DataFrame()
    metrics_df = pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()
    return forecast_df, metrics_df


def build_machine_history(event_log, machine_id):
    if not event_log:
        return pd.DataFrame()
    df = pd.DataFrame(event_log)
    if "Machine_ID" not in df.columns:
        return pd.DataFrame()
    return df[df["Machine_ID"].astype(str) == str(machine_id)].copy()


def render_live_visuals(event, event_log):
    """Render live sensor, health, failure-probability and RUL histories."""
    if event is None:
        return

    hist = build_machine_history(event_log, event.get("Machine_ID"))
    if len(hist) < 2:
        st.info("📈 Process more events to build live graphical history.")
        return

    if "Cycle" in hist.columns:
        hist = hist.sort_values("Cycle")

    st.divider()
    st.subheader("📈 Live Sensor History")

    groups = [
        ("🌡️ Thermal", ["Temperature_C"]),
        ("〰️ Mechanical", ["Vibration_mm_s", "Pressure_bar"]),
        ("⚙️ Machine Operation", ["Rotational_Speed_RPM", "Load_Fraction", "Torque_Nm"]),
        ("⚡ Electrical & Acoustic", ["Current_A", "Acoustic_dB"]),
    ]

    for title, columns in groups:
        available = [c for c in columns if c in hist.columns]
        if not available:
            continue
        st.markdown(f"#### {title}")
        plot_df = hist[available].apply(pd.to_numeric, errors="coerce")
        if "Cycle" in hist.columns:
            plot_df.index = hist["Cycle"].astype(str).values
            plot_df.index.name = "Cycle"
        st.line_chart(plot_df, use_container_width=True)

    st.markdown("#### 🧠 Machine Health Evolution")
    health_cols = [
        c for c in ["failure_probability", "predicted_rul_cycles", "health_score"]
        if c in hist.columns
    ]
    if health_cols:
        health_df = hist[health_cols].apply(pd.to_numeric, errors="coerce")
        if "Cycle" in hist.columns:
            health_df.index = hist["Cycle"].astype(str).values
            health_df.index.name = "Cycle"
        st.line_chart(health_df, use_container_width=True)


def render_time_series_forecast(event):
    """Render the project's actual test predictions and future forecasts."""
    forecast_df, metrics_df = load_forecast_artifacts(ROOT)

    st.divider()
    st.subheader("🔮 Time-Series Forecast")

    if forecast_df.empty:
        st.warning(
            "Forecast artifact not found: "
            "data/processed/time_series_forecasts.csv"
        )
        return

    required = {"Machine_ID", "Sensor", "Cycle", "Prediction_Type"}
    if not required.issubset(forecast_df.columns):
        st.error(f"Unexpected forecast schema. Required: {sorted(required)}")
        return

    machine_id = str(event.get("Machine_ID", ""))
    machine_df = forecast_df[
        forecast_df["Machine_ID"].astype(str) == machine_id
    ].copy()

    if machine_df.empty:
        st.info(f"No forecast records found for {machine_id}.")
        return

    sensors = sorted(machine_df["Sensor"].dropna().astype(str).unique())
    # Keep the operator's sensor selection when switching machines.
    # If the selected sensor is not available on the new machine, fall back
    # to the first available sensor instead of silently resetting the widget.
    if "forecast_sensor" not in st.session_state or st.session_state.forecast_sensor not in sensors:
        st.session_state.forecast_sensor = sensors[0]

    sensor = st.selectbox(
        "Forecast sensor",
        sensors,
        key="forecast_sensor",
    )

    selected = machine_df[machine_df["Sensor"].astype(str) == sensor].copy()
    selected["Cycle"] = pd.to_numeric(selected["Cycle"], errors="coerce")
    selected["Actual_Value"] = pd.to_numeric(
        selected["Actual_Value"], errors="coerce"
    )
    selected["Predicted_Value"] = pd.to_numeric(
        selected["Predicted_Value"], errors="coerce"
    )
    selected = selected.sort_values("Cycle")

    ptype = selected["Prediction_Type"].astype(str).str.upper()
    test = selected[ptype.isin(["TEST", "PREDICTION", "PREDICTED"])].copy()
    future = selected[ptype.isin(["FUTURE_FORECAST", "FUTURE", "FORECAST"])].copy()

    if future.empty:
        future = selected[selected["Actual_Value"].isna()].copy()

    chart = pd.DataFrame()
    if not test.empty:
        chart = test.set_index("Cycle")[["Actual_Value", "Predicted_Value"]].rename(
            columns={
                "Actual_Value": "Actual",
                "Predicted_Value": "Test Prediction",
            }
        )

    if not future.empty:
        future_chart = future.set_index("Cycle")[["Predicted_Value"]].rename(
            columns={"Predicted_Value": "Future Forecast"}
        )
        chart = pd.concat([chart, future_chart], axis=1)

    if not chart.empty:
        st.markdown("**Actual vs Test Prediction + Future Forecast**")
        st.line_chart(chart.sort_index(), use_container_width=True)

    if not metrics_df.empty and {"Machine_ID", "Sensor"}.issubset(metrics_df.columns):
        m = metrics_df[
            (metrics_df["Machine_ID"].astype(str) == machine_id)
            & (metrics_df["Sensor"].astype(str) == sensor)
        ]
        if not m.empty:
            row = m.iloc[0]
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("MAE", format_value(row.get("MAE"), 4))
            with c2:
                st.metric("RMSE", format_value(row.get("RMSE"), 4))
            with c3:
                st.metric("Forecast Rows", f"{len(selected):,}")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Test Predictions", f"{len(test):,}")
    with c2:
        st.metric("Future Forecast", f"{len(future):,}")
    with c3:
        if not future.empty:
            horizon = int(future["Cycle"].max() - future["Cycle"].min() + 1)
            st.metric("Future Horizon", f"{horizon:,} cycles")
        else:
            st.metric("Future Horizon", "—")




# ============================================================
# VISUAL ANALYTICS
# ============================================================

if event:
    render_live_visuals(event, st.session_state.event_log)
    render_time_series_forecast(event)


# ============================================================
# MACHINE INTELLIGENCE DETAILS
# ============================================================

if event:
    st.divider()
    st.subheader("🧠 Machine Intelligence Details")

    tab1, tab2, tab3 = st.tabs([
        "🤖 Decision",
        "🧠 Memory",
        "📚 RAG Evidence",
    ])

    with tab1:
        left, right = st.columns(2)

        with left:
            st.markdown("### Agent Layer")
            invoked = bool(event.get("agents_invoked", False))
            if invoked:
                st.success("Agents INVOKED")
            else:
                st.info("Agents NOT INVOKED")
            st.write(f"**Trigger:** {event.get('trigger_reason')}")
            st.write(f"**Priority:** {event.get('maintenance_priority')}")
            st.write(f"**Decision:** {event.get('maintenance_decision')}")
            st.write(f"**Reason:** {event.get('maintenance_reason')}")

        with right:
            st.markdown("### Alert Layer")
            level = str(event.get("alert_level", "NONE")).upper()
            if level == "CRITICAL":
                st.error(f"🔴 {level}")
            elif level == "WARNING":
                st.warning(f"🟠 {level}")
            else:
                st.success(f"🟢 {level}")
            st.write(f"**Required:** {event.get('alert_required')}")
            st.write(f"**Message:** {event.get('alert_message') or 'No alert'}")

        if st.session_state.llm_report:
            st.markdown("### 💬 Latest Maintenance Intelligence")
            st.markdown(st.session_state.llm_report)

    with tab2:
        machine_history = build_machine_history(
            st.session_state.event_log, event["Machine_ID"]
        )

        if len(machine_history) > 1:
            previous = machine_history.iloc[-2]
            m1, m2, m3, m4 = st.columns(4)

            with m1:
                st.metric("Previous Cycle", str(previous.get("Cycle", "—")))
            with m2:
                current_prob = float(event.get("failure_probability", 0) or 0)
                previous_prob = float(previous.get("failure_probability", 0) or 0)
                st.metric(
                    "Failure Probability Δ",
                    format_value(current_prob - previous_prob, 4),
                )
            with m3:
                current_rul = float(event.get("predicted_rul_cycles", 0) or 0)
                previous_rul = float(previous.get("predicted_rul_cycles", 0) or 0)
                st.metric("RUL Δ", format_value(current_rul - previous_rul, 2))
            with m4:
                changed = previous.get("overall_status") != event.get("overall_status")
                st.metric("Status Changed", "YES" if changed else "NO")

            memory_cols = [
                c for c in [
                    "Cycle", "failure_probability", "predicted_rul_cycles",
                    "health_score", "overall_status", "maintenance_decision"
                ] if c in machine_history.columns
            ]
            st.dataframe(
                machine_history[memory_cols].tail(20),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No previous event for this machine yet.")

    with tab3:
        retrieved = st.session_state.last_retrieved

        rc1, rc2 = st.columns(2)
        with rc1:
            st.metric("Live runtime events in RAG", f"{rag.runtime_event_count:,}")
        with rc2:
            st.metric("Latest RAG results", f"{len(retrieved):,}")

        if retrieved:
            st.caption(
                "RAG combines pre-existing maintenance knowledge with "
                "machine events observed during this live simulation."
            )

            for rank, item in enumerate(retrieved, start=1):
                score = item.get("_retrieval_score", 0)
                source = item.get("document_id")

                if pd.isna(source) or not source:
                    source = item.get("knowledge_id")

                if pd.isna(source) or not source:
                    source = f"row-{item.get('_source_row', rank)}"

                source_type = str(
                    item.get("source_type", "STATIC_MAINTENANCE_KNOWLEDGE")
                )

                label = (
                    "🔴 LIVE MACHINE EVIDENCE"
                    if source_type == "LIVE_RUNTIME_EVENT"
                    else "📘 MAINTENANCE KNOWLEDGE"
                )

                with st.expander(
                    f"{label} · {rank} · {source} · score {score}"
                ):
                    st.write(item.get(rag.text_column, ""))
        else:
            st.info(
                "No RAG retrieval has been performed for the current "
                "actionable event yet."
            )

        # Keep recent retrievals visible even when another machine produces
        # a normal event afterwards.
        rag_history = st.session_state.get("rag_retrieval_history", [])
        if rag_history:
            st.markdown("#### 🕘 Recent Live RAG Retrievals")
            for item in reversed(rag_history[-5:]):
                evt = item.get("event", {})
                results = item.get("retrieved", [])
                st.markdown(
                    f"**{evt.get('Machine_ID', '—')} · Cycle "
                    f"{evt.get('Cycle', '—')} · "
                    f"{evt.get('Simulation_Fault', '—')}** — "
                    f"{len(results)} result(s)"
                )


# ============================================================
# FLOATING MAINTENANCE INTELLIGENCE ASSISTANT
# ============================================================

# st.popover closes on Streamlit reruns. Live playback reruns the app every
# tick, so the assistant uses session state + a fixed container instead.
st.markdown(
    """
    <style>
    /* ---------- Modern operator dashboard ---------- */
    [data-testid="stMetric"] {
        background: rgba(255,255,255,.72) !important;
        border: 1px solid rgba(15,23,42,.08) !important;
        border-radius: 14px !important;
        padding: 12px 14px !important;
        box-shadow: 0 4px 16px rgba(15,23,42,.06) !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px !important;
        padding: 4px !important;
        border-radius: 12px !important;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 9px !important;
        padding: 8px 14px !important;
    }

    /* ---------- Floating assistant ---------- */
    .st-key-chat_launcher {
        position: fixed !important;
        right: 22px !important;
        bottom: 22px !important;
        z-index: 2147483647 !important;
    }
    .st-key-chat_launcher button {
        width: 58px !important;
        height: 58px !important;
        min-width: 58px !important;
        min-height: 58px !important;
        padding: 0 !important;
        border-radius: 50% !important;
        font-size: 24px !important;
        border: 1px solid rgba(15,23,42,.12) !important;
        box-shadow: 0 10px 30px rgba(15,23,42,.22) !important;
        transition: transform .15s ease, box-shadow .15s ease !important;
    }
    .st-key-chat_launcher button:hover {
        transform: translateY(-2px) scale(1.03) !important;
        box-shadow: 0 14px 34px rgba(15,23,42,.28) !important;
    }
    .st-key-floating_chat {
        position: fixed !important;
        right: 22px !important;
        bottom: 94px !important;
        width: 430px !important;
        max-width: calc(100vw - 32px) !important;
        height: 70vh !important;
        max-height: 700px !important;
        min-height: 480px !important;
        z-index: 2147483646 !important;
        background: #ffffff !important;
        color: #111827 !important;
        border: 1px solid rgba(15,23,42,.12) !important;
        border-radius: 20px !important;
        box-shadow: 0 20px 55px rgba(15,23,42,.25) !important;
        padding: 16px !important;
        overflow-y: auto !important;
    }
    .st-key-floating_chat h3 {
        margin-top: 0 !important;
        padding-bottom: 10px !important;
        border-bottom: 1px solid #e5e7eb !important;
    }
    .st-key-floating_chat [data-testid="stForm"] {
        position: sticky !important;
        bottom: -16px !important;
        background: #ffffff !important;
        padding-top: 10px !important;
        padding-bottom: 4px !important;
        border-top: 1px solid #e5e7eb !important;
    }
    .st-key-floating_chat [data-testid="stChatMessage"] {
        border-radius: 12px !important;
        margin-bottom: 8px !important;
    }

    /* ---------- Alert emphasis ---------- */
    .operator-alert-card {
        border-radius: 16px;
        padding: 16px 18px;
        margin: 10px 0 18px 0;
        box-shadow: 0 8px 24px rgba(15,23,42,.10);
        border: 1px solid rgba(15,23,42,.10);
    }
    .operator-alert-critical {
        background: linear-gradient(135deg, #fff1f2, #ffffff);
        border-left: 6px solid #dc2626;
    }
    .operator-alert-warning {
        background: linear-gradient(135deg, #fff7ed, #ffffff);
        border-left: 6px solid #f59e0b;
    }
    .operator-alert-title {
        font-size: 1.05rem;
        font-weight: 750;
        margin-bottom: 6px;
    }
    .operator-alert-meta {
        color: #4b5563;
        font-size: .92rem;
        line-height: 1.55;
    }
    @media (max-width: 600px) {
        .st-key-floating_chat {
            right: 10px !important;
            bottom: 82px !important;
            width: calc(100vw - 20px) !important;
            height: 72vh !important;
            min-height: 420px !important;
        }
        .st-key-chat_launcher {
            right: 12px !important;
            bottom: 12px !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


launcher_label = "✕" if st.session_state.chat_open else "🤖"
if st.button(
    launcher_label,
    key="chat_launcher",
    help=("Close Maintenance Intelligence Assistant"
          if st.session_state.chat_open
          else "Open Maintenance Intelligence Assistant"),
):
    st.session_state.chat_open = not st.session_state.chat_open
    st.rerun()

if st.session_state.chat_open:
    with st.container(key="floating_chat"):
        st.markdown("### 🤖 Maintenance Intelligence Assistant")
        st.caption(
            "Ask about the selected machine. Uses current Smart Twin evidence, "
            "machine memory, live RAG evidence, alert history, and MaaS LLM."
        )
        if not event:
            st.info("Process at least one IoT event first.")
        else:
            if st.session_state.chat_history:
                for role, message in st.session_state.chat_history:
                    with st.chat_message(role):
                        st.markdown(message)
            else:
                st.info(
                    "Ask about the machine's condition, RUL, detected fault, "
                    "maintenance decision, alerts, or recommended action."
                )

            with st.form("floating_maintenance_chat_form", clear_on_submit=True):
                question = st.text_input(
                    "Message",
                    placeholder="e.g. Why should MOTOR_004 be inspected?",
                    label_visibility="collapsed",
                )
                submitted = st.form_submit_button("Send ➤", use_container_width=True)

            if submitted and question.strip():
                try:
                    if st.session_state.llm is None:
                        st.session_state.llm = MaintenanceLLM()
                    machine_context = {
                        key: value for key, value in event.items() if key != "llm_report"
                    }
                    retrieved = rag.retrieve(
                        query=question.strip(),
                        machine_context=machine_context,
                        top_k=3,
                    )
                    rag_context = rag.format_context(retrieved)
                    history = pd.DataFrame(st.session_state.event_log)
                    if not history.empty and "Machine_ID" in history.columns:
                        machine_memory = history[
                            history["Machine_ID"].astype(str) == str(event["Machine_ID"])
                        ].tail(10)
                        memory_context = machine_memory.to_string(index=False)
                    else:
                        memory_context = ""
                    orchestrator_result = {
                        key: event.get(key)
                        for key in [
                            "agents_invoked", "trigger_reason", "maintenance_priority",
                            "maintenance_decision", "maintenance_reason", "alert_level",
                            "alert_required", "alert_message",
                        ]
                    }
                    alert_history_context = get_alert_history_context()
                    st.session_state.chat_history.append(("user", question.strip()))
                    with st.spinner("Retrieving maintenance knowledge and asking MaaS..."):
                        answer = st.session_state.llm.ask(
                            question=question.strip(),
                            machine_context=machine_context,
                            orchestrator_result=orchestrator_result,
                            memory_context=memory_context,
                            rag_context=rag_context,
                            alert_history_context=alert_history_context,
                        )
                    st.session_state.chat_history.append(("assistant", answer))
                    st.session_state.last_retrieved = retrieved
                    st.rerun()
                except Exception as exc:
                    st.error(f"MaaS/RAG request failed: {exc}")


# ============================================================
# EVENT LOG
# ============================================================

st.divider()
st.subheader("📋 Runtime Event Log")

if st.session_state.event_log:
    log_df = pd.DataFrame(
        st.session_state.event_log
    )

    columns = [
        c
        for c in [
            "Cycle",
            "Machine_ID",
            "Simulation_Fault",
            "failure_probability",
            "failure_risk",
            "predicted_rul_cycles",
            "overall_status",
            "agents_invoked",
            "maintenance_decision",
            "alert_level",
        ]
        if c in log_df.columns
    ]

    st.dataframe(
        log_df[columns].tail(30),
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info("No runtime events processed yet.")


# ============================================================
# AUTO PLAYBACK
# ============================================================

if st.session_state.running:
    if not engine.finished:
        time.sleep(float(speed))
        process_one_event()
        st.rerun()
    else:
        st.session_state.running = False
        st.success("🎉 Complete IoT simulation processed.")


# ============================================================
# FOOTER
# ============================================================

progress = (
    engine.event_index / engine.total_events
    if engine.total_events
    else 0
)

st.progress(
    progress,
    text=(
        f"Simulation progress: "
        f"{progress * 100:.1f}% "
        f"({engine.event_index}/{engine.total_events})"
    ),
)

st.caption(
    "LOCAL APPLICATION · Deterministic Smart Twin/Orchestrator decisions "
    "are authoritative; RAG and MaaS LLM provide supporting intelligence."
)
