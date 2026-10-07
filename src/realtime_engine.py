
"""
Local Smart Manufacturing realtime execution engine.

This module extracts the runtime logic already implemented in:
- IoT_Simulation.ipynb
- Smart_Twin_Prediction_Final.ipynb
- 06_Agent_Orchestrator.ipynb

It does NOT retrain models and it does NOT replace the existing notebooks.
It reuses the same trained model artifacts, feature definitions, thresholds,
maintenance rules, and state-transition rules for application-time execution.

Current simulation source:
    data/iot/iot_simulation_data.csv

Runtime flow:
    IoT event
      -> Smart Twin prediction
      -> Agent/orchestrator decision
      -> state transition tracking

Memory/RAG/LLM remain separate application layers and can consume the
returned runtime event without changing the authoritative ML/orchestrator
decision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, Any

import joblib
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

def find_project_root(start: Optional[Path] = None) -> Path:
    """Locate the repository root from the current working directory."""
    start = (start or Path.cwd()).resolve()

    candidates = [start] + list(start.parents)

    for root in candidates:
        if (
            (root / "data" / "iot").exists()
            and (root / "data" / "processed").exists()
            and (root / "models").exists()
        ):
            return root

    raise FileNotFoundError(
        "Could not locate the project root. "
        "Run the application from the repository or pass project_root."
    )


# ============================================================
# EXACT FEATURE DEFINITIONS FROM SMART TWIN NOTEBOOK
# ============================================================

FAILURE_FEATURES = [
    "Cycle",
    "Temperature_C",
    "Vibration_mm_s",
    "Pressure_bar",
    "Rotational_Speed_RPM",
    "Load_Fraction",
    "Torque_Nm",
    "Current_A",
    "Acoustic_dB",
    "Wear_Level",
    "Machine_Type",
]

RUL_FEATURES = [
    "Cycle",
    "Temperature_C",
    "Vibration_mm_s",
    "Pressure_bar",
    "Rotational_Speed_RPM",
    "Load_Fraction",
    "Torque_Nm",
    "Current_A",
    "Acoustic_dB",
    "Wear_Level",
    "Health_Index",
    "Anomaly_Score",
    "Machine_Type",
]


# ============================================================
# EXACT SMART TWIN RULES FROM NOTEBOOK
# ============================================================

def get_maintenance_recommendation(
    failure_risk: str,
    predicted_rul: float,
) -> str:
    if failure_risk == "HIGH" or predicted_rul <= 20:
        return "IMMEDIATE MAINTENANCE"

    elif failure_risk == "MEDIUM" or predicted_rul <= 50:
        return "SCHEDULE INSPECTION"

    else:
        return "NORMAL OPERATION"


def calculate_health_score(
    failure_probability: float,
    predicted_rul: float,
) -> float:
    failure_score = (1 - failure_probability) * 100
    rul_score = min(predicted_rul, 100)

    health_score = (
        0.6 * failure_score
        + 0.4 * rul_score
    )

    return round(health_score, 2)


# ============================================================
# SMART TWIN ENGINE
# ============================================================

class SmartTwinEngine:
    """Loads the project's existing trained models and performs inference."""

    def __init__(
        self,
        failure_model,
        rul_model,
    ):
        self.failure_model = failure_model
        self.rul_model = rul_model

    @classmethod
    def from_project(cls, project_root: Optional[Path] = None):
        root = project_root or find_project_root()

        failure_path = root / "models" / "failure_prediction_model.pkl"
        rul_path = root / "models" / "rul_prediction_model.pkl"

        if not failure_path.exists():
            raise FileNotFoundError(
                f"Failure model not found: {failure_path}"
            )

        if not rul_path.exists():
            raise FileNotFoundError(
                f"RUL model not found: {rul_path}"
            )

        return cls(
            failure_model=joblib.load(failure_path),
            rul_model=joblib.load(rul_path),
        )

    def predict_machine_state(self, machine_data: Dict[str, Any]) -> Dict[str, Any]:
        machine_df = pd.DataFrame([machine_data])

        missing_failure = [
            c for c in FAILURE_FEATURES
            if c not in machine_df.columns
        ]
        missing_rul = [
            c for c in RUL_FEATURES
            if c not in machine_df.columns
        ]

        if missing_failure or missing_rul:
            raise ValueError(
                "Machine event is missing required model features. "
                f"Failure missing={missing_failure}; "
                f"RUL missing={missing_rul}"
            )

        failure_input = machine_df[FAILURE_FEATURES]

        failure_prediction = self.failure_model.predict(
            failure_input
        )[0]

        failure_probability = self.failure_model.predict_proba(
            failure_input
        )[0][1]

        rul_input = machine_df[RUL_FEATURES]

        predicted_rul = self.rul_model.predict(
            rul_input
        )[0]

        predicted_rul = max(0, float(predicted_rul))

        if failure_probability >= 0.70:
            failure_risk = "HIGH"
        elif failure_probability >= 0.30:
            failure_risk = "MEDIUM"
        else:
            failure_risk = "LOW"

        if predicted_rul <= 20:
            rul_status = "CRITICAL"
        elif predicted_rul <= 50:
            rul_status = "WARNING"
        else:
            rul_status = "HEALTHY"

        if (
            failure_risk == "HIGH"
            or rul_status == "CRITICAL"
        ):
            overall_status = "CRITICAL"
        elif (
            failure_risk == "MEDIUM"
            or rul_status == "WARNING"
        ):
            overall_status = "WARNING"
        else:
            overall_status = "HEALTHY"

        maintenance_action = get_maintenance_recommendation(
            failure_risk,
            predicted_rul,
        )

        health_score = calculate_health_score(
            failure_probability,
            predicted_rul,
        )

        return {
            "failure_prediction": int(failure_prediction),
            "failure_probability": round(
                float(failure_probability), 4
            ),
            "failure_risk": failure_risk,
            "predicted_rul_cycles": round(
                predicted_rul, 2
            ),
            "rul_status": rul_status,
            "overall_status": overall_status,
            "maintenance_action": maintenance_action,
            "health_score": health_score,
        }


# ============================================================
# EXACT AGENT RULES FROM 06_Agent_Orchestrator.ipynb
# ============================================================

def maintenance_agent(row: Dict[str, Any]) -> Dict[str, Any]:
    failure_risk = row["failure_risk"]
    rul = float(row["predicted_rul_cycles"])
    status = row["overall_status"]
    health = float(row["health_score"])

    if (
        failure_risk == "HIGH"
        or rul <= 20
        or status == "CRITICAL"
        or health < 30
    ):
        return {
            "maintenance_priority": "URGENT",
            "maintenance_decision": "IMMEDIATE MAINTENANCE",
            "maintenance_reason": (
                "Critical machine condition detected."
            ),
        }

    elif (
        failure_risk == "MEDIUM"
        or rul <= 50
        or status == "WARNING"
    ):
        return {
            "maintenance_priority": "HIGH",
            "maintenance_decision": "SCHEDULE INSPECTION",
            "maintenance_reason": (
                "Machine degradation requires inspection."
            ),
        }

    else:
        return {
            "maintenance_priority": "ROUTINE",
            "maintenance_decision": "NORMAL OPERATION",
            "maintenance_reason": (
                "Machine is operating within the healthy range."
            ),
        }


def alert_agent(
    row: Dict[str, Any],
    maintenance_result: Dict[str, Any],
) -> Dict[str, Any]:
    probability = float(row["failure_probability"])
    rul = float(row["predicted_rul_cycles"])

    risk = row["failure_risk"]
    status = row["overall_status"]

    machine_id = row["Machine_ID"]
    machine_type = row["Machine_Type"]
    cycle = row["Cycle"]

    action = maintenance_result["maintenance_decision"]

    if (
        risk == "HIGH"
        or rul <= 20
        or status == "CRITICAL"
    ):
        level = "CRITICAL"
        required = True

        message = (
            f"CRITICAL ALERT | {machine_id} "
            f"({machine_type}) | Cycle {cycle} | "
            f"Failure probability={probability:.2%} | "
            f"RUL={rul:.2f} cycles | "
            f"Action={action}"
        )

    elif (
        risk == "MEDIUM"
        or rul <= 50
        or status == "WARNING"
    ):
        level = "WARNING"
        required = True

        message = (
            f"WARNING | {machine_id} "
            f"({machine_type}) | Cycle {cycle} | "
            f"Failure probability={probability:.2%} | "
            f"RUL={rul:.2f} cycles | "
            f"Action={action}"
        )

    else:
        level = "INFO"
        required = False

        message = (
            f"INFO | {machine_id} | "
            "Operating normally."
        )

    return {
        "alert_required": required,
        "alert_level": level,
        "alert_message": message,
    }


def should_trigger_agents(
    row: Dict[str, Any],
    previous_state: Optional[Dict[str, Any]],
) -> Tuple[bool, str]:
    risk = row["failure_risk"]
    status = row["overall_status"]
    rul = float(row["predicted_rul_cycles"])

    actionable_now = (
        risk in ["MEDIUM", "HIGH"]
        or status in ["WARNING", "CRITICAL"]
        or rul <= 50
    )

    if not actionable_now:
        return False, "NORMAL"

    if previous_state is None:
        return True, "NEW_ACTIONABLE_STATE"

    previous_risk = previous_state["failure_risk"]
    previous_status = previous_state["overall_status"]
    previous_rul = float(
        previous_state["predicted_rul_cycles"]
    )

    risk_changed = previous_risk != risk
    status_changed = previous_status != status

    rul_crossed = (
        previous_rul > 50
        and rul <= 50
    )

    critical_crossed = (
        previous_rul > 20
        and rul <= 20
    )

    if (
        risk_changed
        or status_changed
        or rul_crossed
        or critical_crossed
    ):
        return True, "STATE_CHANGE"

    return False, "ALREADY_ACTIONABLE"


def run_orchestrator(
    row: Dict[str, Any],
    previous_state: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    trigger, trigger_reason = should_trigger_agents(
        row,
        previous_state,
    )

    if not trigger:
        return {
            "agents_invoked": False,
            "trigger_reason": trigger_reason,
            "maintenance_priority": "NONE",
            "maintenance_decision": "NO ACTION",
            "alert_level": "NONE",
            "alert_required": False,
            "alert_message": "",
        }

    maintenance_result = maintenance_agent(row)
    alert_result = alert_agent(
        row,
        maintenance_result,
    )

    return {
        "agents_invoked": True,
        "trigger_reason": trigger_reason,
        **maintenance_result,
        **alert_result,
    }


# ============================================================
# RUNTIME ENGINE
# ============================================================

@dataclass
class RuntimeEngine:
    """
    Stateful cycle-by-cycle application engine.

    The engine follows the same parallel-machine ordering used by
    IoT_Simulation.ipynb and the same previous-state transition logic
    used by 06_Agent_Orchestrator.ipynb.
    """

    project_root: Path
    iot_df: pd.DataFrame
    smart_twin: SmartTwinEngine
    machine_state: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    event_index: int = 0
    events: List[Dict[str, Any]] = field(default_factory=list)

    @classmethod
    def create(cls, project_root: Optional[Path] = None):
        root = project_root or find_project_root()

        iot_path = root / "data" / "iot" / "iot_simulation_data.csv"

        if not iot_path.exists():
            raise FileNotFoundError(
                f"IoT simulation file not found: {iot_path}"
            )

        df = pd.read_csv(iot_path)

        required = set(FAILURE_FEATURES + RUL_FEATURES)
        required.update(
            [
                "Machine_ID",
                "Machine_Type",
                "Timestamp",
                "Simulation_Fault",
            ]
        )

        missing = sorted(required - set(df.columns))

        if missing:
            raise ValueError(
                "IoT dataset is missing required columns: "
                + ", ".join(missing)
            )

        df["Machine_ID"] = df["Machine_ID"].astype(str)
        df["Cycle"] = pd.to_numeric(
            df["Cycle"], errors="coerce"
        )
        df = df.dropna(subset=["Cycle"]).copy()
        df["Cycle"] = df["Cycle"].astype(int)

        # Preserve the notebook's machine-wise ordered streams and
        # cycle-by-cycle parallel simulation ordering.
        df = df.sort_values(
            ["Cycle", "Machine_ID"]
        ).reset_index(drop=True)

        events = [
            row.to_dict()
            for _, row in df.iterrows()
        ]

        return cls(
            project_root=root,
            iot_df=df,
            smart_twin=SmartTwinEngine.from_project(root),
            events=events,
        )

    @property
    def total_events(self) -> int:
        return len(self.events)

    @property
    def finished(self) -> bool:
        return self.event_index >= len(self.events)

    def reset(self) -> None:
        self.machine_state.clear()
        self.event_index = 0

    def process_event(
        self,
        machine_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Execute one event through Smart Twin + orchestrator.

        The returned event contains the raw IoT data plus authoritative
        Smart Twin and orchestrator results.
        """

        smart_twin_result = (
            self.smart_twin.predict_machine_state(
                machine_data
            )
        )

        row = {
            **machine_data,
            **smart_twin_result,
        }

        machine_id = str(row["Machine_ID"])
        previous_state = self.machine_state.get(machine_id)

        orchestrator_result = run_orchestrator(
            row,
            previous_state,
        )

        # Preserve current state for transition detection exactly as
        # done in the notebook.
        self.machine_state[machine_id] = {
            "failure_risk": row["failure_risk"],
            "overall_status": row["overall_status"],
            "predicted_rul_cycles": row[
                "predicted_rul_cycles"
            ],
        }

        return {
            **row,
            **orchestrator_result,
        }

    def next_event(self) -> Optional[Dict[str, Any]]:
        """
        Process exactly one IoT event.

        Returns None after the complete simulated stream has been consumed.
        """
        if self.finished:
            return None

        raw_event = self.events[self.event_index]
        result = self.process_event(raw_event)

        self.event_index += 1
        return result

    def run_cycle(self, cycle: int) -> List[Dict[str, Any]]:
        """
        Process all machines belonging to one simulation cycle.

        This mirrors the notebook's:
            for cycle in available_cycles:
                cycle_rows = stream_df[stream_df["Cycle"] == cycle]
        """
        cycle_rows = self.iot_df[
            self.iot_df["Cycle"] == cycle
        ]

        results = []

        for _, row in cycle_rows.iterrows():
            results.append(
                self.process_event(
                    row.to_dict()
                )
            )

        # Advance the event cursor past events in this cycle.
        while (
            self.event_index < len(self.events)
            and int(self.events[self.event_index]["Cycle"])
            == int(cycle)
        ):
            self.event_index += 1

        return results

    def iter_cycles(
        self,
        max_cycles: Optional[int] = None,
    ) -> Iterator[Tuple[int, List[Dict[str, Any]]]]:
        """Yield the simulation one cycle at a time."""
        cycles = sorted(
            self.iot_df["Cycle"].unique()
        )

        if max_cycles is not None:
            cycles = cycles[:max_cycles]

        for cycle in cycles:
            yield int(cycle), self.run_cycle(int(cycle))

    def run_all(
        self,
        max_cycles: Optional[int] = None,
    ) -> pd.DataFrame:
        """Run the complete local simulation through the engine."""
        results = []

        for _, cycle_results in self.iter_cycles(
            max_cycles=max_cycles
        ):
            results.extend(cycle_results)

        return pd.DataFrame(results)


# ============================================================
# SIMPLE CLI TEST
# ============================================================

if __name__ == "__main__":
    engine = RuntimeEngine.create()

    print("Realtime engine initialized.")
    print("Project:", engine.project_root)
    print("Total IoT events:", engine.total_events)

    first = engine.next_event()

    if first is not None:
        print("\nFIRST RUNTIME EVENT")
        print("===================")

        for key in [
            "Machine_ID",
            "Machine_Type",
            "Cycle",
            "Simulation_Fault",
            "failure_probability",
            "failure_risk",
            "predicted_rul_cycles",
            "rul_status",
            "overall_status",
            "maintenance_action",
            "health_score",
            "agents_invoked",
            "trigger_reason",
            "maintenance_priority",
            "maintenance_decision",
            "alert_level",
            "alert_required",
        ]:
            print(f"{key}: {first.get(key)}")
