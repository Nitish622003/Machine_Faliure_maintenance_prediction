
"""
TCS MaaS LLM application layer.

The LLM receives authoritative Smart Twin/orchestrator evidence plus the
RAG-retrieved maintenance knowledge. It produces an explanation/report; it
does not make or override the deterministic maintenance decision.

The connection configuration matches the validated project setup:
    base_url = https://genailab.tcs.in
    model = genailab-maas-gpt-35-turbo
    httpx.Client(verify=False)
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import httpx
from langchain_openai import ChatOpenAI


class MaintenanceLLM:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "genailab-maas-gpt-35-turbo",
        base_url: str = "https://genailab.tcs.in",
        temperature: float = 0.2,
    ):
        self.api_key = api_key or os.getenv(
            "GENAI_API_KEY"
        )

        if not self.api_key:
            raise RuntimeError(
                "GENAI_API_KEY is not available. "
                "Export it in the same terminal used to launch Streamlit."
            )

        self.client = ChatOpenAI(
            base_url=base_url,
            model=model,
            api_key=self.api_key,
            http_client=httpx.Client(
                verify=False
            ),
            temperature=temperature,
        )

    @staticmethod
    def build_prompt(
        question: str,
        machine_context: Dict[str, Any],
        orchestrator_result: Optional[Dict[str, Any]] = None,
        memory_context: Optional[str] = None,
        rag_context: Optional[str] = None,
        alert_history_context: Optional[str] = None,
    ) -> str:
        machine_lines = "\n".join(
            f"- {key}: {value}"
            for key, value in machine_context.items()
        )

        orchestrator_lines = ""

        if orchestrator_result:
            orchestrator_lines = "\n".join(
                f"- {key}: {value}"
                for key, value in orchestrator_result.items()
            )

        return f"""
You are the Smart Manufacturing Maintenance Intelligence Assistant.

Your role is to explain the machine's condition and maintenance implications
using authoritative project evidence.

IMPORTANT RULES:
1. Smart Twin and orchestrator decisions are authoritative.
2. Do not invent sensor values, predictions, events, or maintenance actions.
3. Do not override the deterministic maintenance decision.
4. Clearly distinguish observed sensor values from model predictions.
5. Use RAG knowledge only as supporting maintenance knowledge.
6. If evidence is insufficient, explicitly say so.
7. Give a concise but technically useful explanation for an engineer or
   maintenance operator.

USER QUESTION:
{question}

CURRENT MACHINE EVIDENCE:
{machine_lines}

ORCHESTRATOR / AGENT DECISION:
{orchestrator_lines or "No separate orchestrator record supplied."}

PREVIOUS MACHINE MEMORY:
{memory_context or "No previous memory supplied."}

ALERT & FAILURE HISTORY:
{alert_history_context or "No alert/failure history supplied."}

RETRIEVED MAINTENANCE KNOWLEDGE:
{rag_context or "No RAG knowledge supplied."}

Return:
- Current condition
- Evidence behind the condition
- Maintenance implication
- Relevant supporting knowledge
- Any uncertainty or limitation
"""

    def ask(
        self,
        question: str,
        machine_context: Dict[str, Any],
        orchestrator_result: Optional[Dict[str, Any]] = None,
        memory_context: Optional[str] = None,
        rag_context: Optional[str] = None,
        alert_history_context: Optional[str] = None,
    ) -> str:
        prompt = self.build_prompt(
            question=question,
            machine_context=machine_context,
            orchestrator_result=orchestrator_result,
            memory_context=memory_context,
            rag_context=rag_context,
            alert_history_context=alert_history_context,
        )

        response = self.client.invoke(prompt)

        return response.content

    def generate_maintenance_report(
        self,
        machine_context: Dict[str, Any],
        orchestrator_result: Optional[Dict[str, Any]] = None,
        memory_context: Optional[str] = None,
        rag_context: Optional[str] = None,
        alert_history_context: Optional[str] = None,
    ) -> str:
        question = (
            "Generate a maintenance intelligence report for the current "
            "machine event. Explain the condition, evidence, maintenance "
            "priority/decision, and supporting maintenance knowledge."
        )

        return self.ask(
            question=question,
            machine_context=machine_context,
            orchestrator_result=orchestrator_result,
            memory_context=memory_context,
            rag_context=rag_context,
            alert_history_context=alert_history_context,
        )


if __name__ == "__main__":
    llm = MaintenanceLLM()

    result = llm.ask(
        question="Why is the machine degrading?",
        machine_context={
            "Machine_ID": "CNC_001",
            "Machine_Type": "CNC_Mill",
            "Cycle": 1,
            "failure_probability": 0.0001,
            "failure_risk": "LOW",
            "predicted_rul_cycles": 256.45,
            "overall_status": "HEALTHY",
        },
        orchestrator_result={
            "agents_invoked": False,
            "trigger_reason": "NORMAL",
            "maintenance_decision": "NO ACTION",
            "alert_level": "NONE",
        },
    )

    print(result)
