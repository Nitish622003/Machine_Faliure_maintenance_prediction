
"""
Local maintenance RAG engine.

This module uses the project's existing RAG maintenance knowledge artifacts.
It is deliberately lightweight: the authoritative machine decision comes
from realtime_engine.py; this module only retrieves supporting maintenance
knowledge.

Expected artifact:
    data/processed/rag_maintenance_reports.csv

The implementation supports several possible text-column layouts so that it
can work with the project's existing generated RAG CSV without changing it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


TEXT_COLUMNS = [
    "maintenance_knowledge",
    "knowledge_text",
    "document_text",
    "content",
    "text",
    "rag_report",
    "llm_report",
    "maintenance_reason",
    "alert_message",
]

ID_COLUMNS = [
    "document_id",
    "knowledge_id",
    "report_id",
    "memory_id",
]


def find_project_root(start: Optional[Path] = None) -> Path:
    start = (start or Path.cwd()).resolve()

    for root in [start] + list(start.parents):
        if (
            (root / "data" / "processed").exists()
            and (root / "data" / "iot").exists()
        ):
            return root

    raise FileNotFoundError("Could not locate the project root.")


class MaintenanceRAG:
    """
    Retrieval over the project's generated maintenance knowledge.

    Retrieval is intentionally deterministic and dependency-light. It uses
    token overlap against the supplied machine/event evidence and question.
    This keeps the RAG layer local and reproducible.
    """

    def __init__(
        self,
        knowledge_df: pd.DataFrame,
        top_k: int = 3,
    ):
        self.knowledge_df = knowledge_df.copy()
        self.top_k = top_k

        # Static maintenance knowledge is loaded from the project artifact.
        # Runtime machine events are appended in-memory during the live
        # simulation; they are never written back to the static CSV.
        if "source_type" not in self.knowledge_df.columns:
            self.knowledge_df["source_type"] = "STATIC_MAINTENANCE_KNOWLEDGE"

        if self.knowledge_df.empty:
            raise ValueError("RAG knowledge dataset is empty.")

        self.text_column = self._find_text_column()

    @classmethod
    def from_project(
        cls,
        project_root: Optional[Path] = None,
        top_k: int = 3,
    ):
        root = project_root or find_project_root()
        path = (
            root
            / "data"
            / "processed"
            / "rag_maintenance_reports.csv"
        )

        if not path.exists():
            raise FileNotFoundError(
                f"RAG artifact not found: {path}"
            )

        df = pd.read_csv(path)

        return cls(df, top_k=top_k)


    def add_runtime_event(self, event: Dict[str, Any]) -> None:
        """Add one processed machine event to the live RAG corpus.

        This creates a searchable runtime evidence record from the authoritative
        Smart Twin/orchestrator output. It deliberately stays in memory so the
        pre-generated maintenance knowledge CSV remains unchanged.
        """
        if not event:
            return

        machine_id = str(event.get("Machine_ID", "UNKNOWN"))
        cycle = event.get("Cycle", "UNKNOWN")
        fault = str(event.get("Simulation_Fault", "NONE"))
        risk = str(event.get("failure_risk", "UNKNOWN"))
        status = str(event.get("overall_status", "UNKNOWN"))
        rul = event.get("predicted_rul_cycles", "UNKNOWN")
        probability = event.get("failure_probability", "UNKNOWN")
        decision = str(event.get("maintenance_decision", "NO ACTION"))
        alert = str(event.get("alert_level", "NONE"))
        trigger = str(event.get("trigger_reason", "NORMAL"))

        text = (
            f"LIVE MACHINE EVENT | Machine_ID={machine_id} | Cycle={cycle} | "
            f"Simulation_Fault={fault} | failure_risk={risk} | "
            f"overall_status={status} | predicted_rul_cycles={rul} | "
            f"failure_probability={probability} | "
            f"maintenance_decision={decision} | alert_level={alert} | "
            f"trigger_reason={trigger}. "
            f"This is runtime machine evidence observed during the current "
            f"simulation and must be treated as observed project data, not as "
            f"general maintenance guidance."
        )

        record = {
            "document_id": f"LIVE_{machine_id}_CYCLE_{cycle}",
            "source_type": "LIVE_RUNTIME_EVENT",
            self.text_column: text,
            "Machine_ID": machine_id,
            "Cycle": cycle,
            "Simulation_Fault": fault,
            "failure_risk": risk,
            "overall_status": status,
            "predicted_rul_cycles": rul,
            "failure_probability": probability,
            "maintenance_decision": decision,
            "alert_level": alert,
        }

        # Replace an existing record for the same machine/cycle if the method
        # is accidentally called twice.
        mask = (
            self.knowledge_df.get("source_type", pd.Series(index=self.knowledge_df.index))
            .astype(str).eq("LIVE_RUNTIME_EVENT")
            & self.knowledge_df.get("Machine_ID", pd.Series(index=self.knowledge_df.index)).astype(str).eq(machine_id)
            & self.knowledge_df.get("Cycle", pd.Series(index=self.knowledge_df.index)).astype(str).eq(str(cycle))
        )
        if mask.any():
            self.knowledge_df.loc[mask, list(record.keys())] = pd.Series(record)
        else:
            self.knowledge_df = pd.concat(
                [self.knowledge_df, pd.DataFrame([record])],
                ignore_index=True,
            )

    def clear_runtime_events(self) -> None:
        """Remove only live runtime records; keep static RAG knowledge."""
        if "source_type" in self.knowledge_df.columns:
            self.knowledge_df = self.knowledge_df[
                self.knowledge_df["source_type"].astype(str)
                != "LIVE_RUNTIME_EVENT"
            ].reset_index(drop=True)

    @property
    def runtime_event_count(self) -> int:
        if "source_type" not in self.knowledge_df.columns:
            return 0
        return int(
            (self.knowledge_df["source_type"].astype(str) == "LIVE_RUNTIME_EVENT").sum()
        )

    def _find_text_column(self) -> str:
        for column in TEXT_COLUMNS:
            if column in self.knowledge_df.columns:
                return column

        # If no conventional text field exists, build a searchable
        # representation from all object/string columns.
        object_columns = [
            c
            for c in self.knowledge_df.columns
            if self.knowledge_df[c].dtype == "object"
        ]

        if not object_columns:
            raise ValueError(
                "No text-like columns were found in the RAG dataset."
            )

        self.knowledge_df["_combined_text"] = (
            self.knowledge_df[object_columns]
            .fillna("")
            .astype(str)
            .agg(" ".join, axis=1)
        )

        return "_combined_text"

    @staticmethod
    def _tokens(text: Any) -> set[str]:
        if text is None:
            return set()

        cleaned = (
            str(text)
            .lower()
            .replace("_", " ")
            .replace("-", " ")
        )

        return {
            token.strip(".,:;!?()[]{}\"'")
            for token in cleaned.split()
            if len(token.strip(".,:;!?()[]{}\"'")) >= 3
        }

    def retrieve(
        self,
        query: str,
        machine_context: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Return the most relevant maintenance knowledge records.

        Ranking is based on token overlap between the query/context and each
        knowledge record. Machine-specific fields receive an additional
        relevance bonus when present.
        """
        k = top_k or self.top_k

        context_parts = [query]

        if machine_context:
            for key, value in machine_context.items():
                if value is not None:
                    context_parts.append(
                        f"{key} {value}"
                    )

        query_tokens = self._tokens(
            " ".join(context_parts)
        )

        scored = []

        for idx, row in self.knowledge_df.iterrows():
            text = str(row[self.text_column])
            text_tokens = self._tokens(text)

            overlap = len(
                query_tokens.intersection(text_tokens)
            )

            score = float(overlap)

            # Add a stronger bonus for live evidence from the same machine.
            # Static maintenance knowledge remains useful, but live runtime
            # records should win when the question is about this machine's history.
            source_type = str(row.get("source_type", "STATIC_MAINTENANCE_KNOWLEDGE"))
            if machine_context and source_type == "LIVE_RUNTIME_EVENT":
                current_machine = machine_context.get("Machine_ID")
                if current_machine is not None and str(row.get("Machine_ID", "")) == str(current_machine):
                    score += 3.0

                current_fault = machine_context.get("Simulation_Fault")
                if current_fault is not None and str(row.get("Simulation_Fault", "")) == str(current_fault):
                    score += 2.0

            # Add a small bonus for exact machine/fault/risk references.
            if machine_context:
                for key in [
                    "Machine_ID",
                    "Machine_Type",
                    "Simulation_Fault",
                    "failure_risk",
                    "overall_status",
                    "maintenance_decision",
                ]:
                    value = machine_context.get(key)

                    if value is not None:
                        value_tokens = self._tokens(value)

                        if value_tokens.intersection(text_tokens):
                            score += 0.5

            scored.append(
                (score, idx, row)
            )

        # IMPORTANT: do not let static maintenance knowledge crowd out the
        # live machine evidence.  For the live dashboard/chatbot, the current
        # runtime event is authoritative and must be available to the LLM.
        scored.sort(
            key=lambda x: (-x[0], x[1])
        )

        def make_item(score, idx, row):
            item = row.to_dict()
            item["_retrieval_score"] = round(float(score), 4)
            item["_source_row"] = int(idx)
            return item

        live_scored = []
        static_scored = []
        for score, idx, row in scored:
            source_type = str(row.get("source_type", "STATIC_MAINTENANCE_KNOWLEDGE")).upper()
            if source_type == "LIVE_RUNTIME_EVENT":
                live_scored.append((score, idx, row))
            else:
                static_scored.append((score, idx, row))

        results = []
        used_rows = set()

        # 1) Always include the current live event for the selected machine.
        # This prevents a healthy/static maintenance document from replacing
        # the actual event that triggered the RAG/LLM request.
        current_live = None
        if machine_context:
            wanted_machine = str(machine_context.get("Machine_ID", ""))
            wanted_cycle = machine_context.get("Cycle", "")

            def same_cycle(a, b):
                try:
                    return float(a) == float(b)
                except (TypeError, ValueError):
                    return str(a) == str(b)

            for candidate in live_scored:
                score, idx, row = candidate
                if (
                    str(row.get("Machine_ID", "")) == wanted_machine
                    and same_cycle(row.get("Cycle", ""), wanted_cycle)
                ):
                    current_live = candidate
                    break

        if current_live is not None:
            score, idx, row = current_live
            results.append(make_item(score, idx, row))
            used_rows.add(idx)

        # 2) Include the strongest relevant static maintenance knowledge.
        if len(results) < k and static_scored:
            score, idx, row = static_scored[0]
            if idx not in used_rows:
                results.append(make_item(score, idx, row))
                used_rows.add(idx)

        # 3) Include the strongest relevant historical runtime event.
        if len(results) < k:
            for score, idx, row in live_scored:
                if idx not in used_rows:
                    results.append(make_item(score, idx, row))
                    used_rows.add(idx)
                    break

        # 4) Fill any remaining slots with the globally highest-scoring
        # records, while avoiding duplicates.
        if len(results) < k:
            for score, idx, row in scored:
                if idx not in used_rows:
                    results.append(make_item(score, idx, row))
                    used_rows.add(idx)
                if len(results) >= k:
                    break

        return results

    def format_context(
        self,
        results: List[Dict[str, Any]],
    ) -> str:
        """Convert retrieved records into an LLM-friendly context."""
        if not results:
            return "No maintenance knowledge was retrieved."

        blocks = []

        for rank, result in enumerate(results, start=1):
            source = next(
                (
                    result.get(column)
                    for column in ID_COLUMNS
                    if result.get(column) is not None
                ),
                f"row-{result.get('_source_row')}",
            )

            score = result.get(
                "_retrieval_score",
                0,
            )

            text = result.get(
                self.text_column,
                "",
            )

            blocks.append(
                f"[Knowledge {rank}] "
                f"Source={source}; "
                f"Type={result.get('source_type', 'STATIC_MAINTENANCE_KNOWLEDGE')}; "
                f"score={score}\n{text}"
            )

        return "\n\n".join(blocks)


if __name__ == "__main__":
    rag = MaintenanceRAG.from_project()

    results = rag.retrieve(
        query="machine degradation bearing vibration maintenance",
        machine_context={
            "Machine_Type": "CNC_Mill",
            "failure_risk": "HIGH",
            "overall_status": "CRITICAL",
        },
    )

    print("RAG engine initialized.")
    print(f"Knowledge records: {len(rag.knowledge_df)}")
    print(f"Text column: {rag.text_column}")
    print(f"Retrieved: {len(results)}")

    print("\nRETRIEVED KNOWLEDGE")
    print("===================")
    print(rag.format_context(results))
