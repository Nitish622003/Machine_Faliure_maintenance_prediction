
# Local Smart Manufacturing Dashboard

## Purpose

This is the first UI layer for the Smart Manufacturing Digital Twin project.

It is intentionally **local-only**. It does not require public hosting.

The dashboard reads the existing project artifacts and presents:

- IoT telemetry
- machine health
- failure probability
- RUL
- maintenance priority
- alert state
- time-series forecasts
- agent/orchestrator activity
- LLM maintenance reports
- machine memory (when the artifact is present)
- RAG evidence/reports (when the artifact is present)
- model monitoring
- an optional live TCS MaaS RAG assistant

## Place the file

Copy:

```text
app/dashboard.py
```

into the root of the project repository.

The expected project structure is:

```text
3203141-Capstone_Proj/
├── app/
│   └── dashboard.py
├── data/
│   ├── iot/
│   │   └── iot_simulation_data.csv
│   ├── processed/
│   │   ├── ai_agent_results.csv
│   │   ├── orchestrated_runtime_events.csv
│   │   ├── llm_maintenance_reports.csv
│   │   ├── time_series_forecasts.csv
│   │   └── model_monitoring_summary.csv
│   └── knowledge/
│       └── maintenance_knowledge_base.json
└── notebooks/
    └── ...
```

## Install UI dependencies

Inside the project's virtual environment:

```bash
pip install streamlit plotly
```

The existing project already uses pandas, numpy, scikit-learn and langchain-openai.

If any of those are missing:

```bash
pip install pandas numpy scikit-learn scipy langchain-openai
```

## Start the dashboard

From the **project root**:

```bash
streamlit run app/dashboard.py
```

The dashboard is local. Streamlit normally opens a browser tab automatically.

## MaaS / RAG chat

For the live RAG assistant, the terminal running Streamlit must have:

```bash
export GENAI_API_KEY='YOUR_KEY'
```

Then:

```bash
streamlit run app/dashboard.py
```

Do not put the real key into `dashboard.py` or commit it to Git.

## Current UI behavior

The first UI version uses the already-generated IoT CSV as a deterministic playback stream.

Use:

- **Simulation Cycle** slider
- **Next** button
- **Advance One Cycle**
- **Reset**

to move through the simulation.

This is intentional. We first validate the presentation layer against known project outputs.

## Next UI iteration

After this version is validated, the dashboard will be upgraded so that one simulated event actually triggers:

```text
IoT event
  ↓
Smart Twin prediction
  ↓
Time-series forecast
  ↓
Agent decision
  ↓
Orchestrator
  ↓
Machine Memory
  ↓
RAG retrieval
  ↓
TCS MaaS LLM
  ↓
Dashboard update
```

That will make the final demo genuinely event-driven rather than merely replaying saved CSVs.
