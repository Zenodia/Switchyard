Advanced Orchestrator Demo

This demo extends `orchestrator_demo.py` with dynamic routing and cascade/failover.

What it shows:
- Dynamic routing: a lightweight classifier asks a cheap profile which logical
~/.venv-py312/bin/python examples/orchestrator_advanced_demo.py
- Cascade/failover: the orchestrator tries the primary profile and falls back
  through a configured list if the response is missing, too short, or an error
switchyard serve --config examples/nvidia_profile.yaml --port 4001 &
export SWITCHYARD_PORT=4001
# Run with Python 3.12 venv to match the installed Switchyard runtime
~/.venv-py312/bin/python examples/orchestrator_advanced_demo.py

Files:
- `orchestrator_advanced_demo.py` — script implementing classifier + cascade.

Run:
1) Start the switchyard server in the background or let the script start it:

```bash
# (option A) let the script start a server on an unused port
~/.venv-py312/bin/python examples/orchestrator_advanced_demo.py

# (option B) start server manually and run script against it
~/.venv-py312/bin/switchyard serve --config examples/nvidia_profile.yaml --port 4001 &
export SWITCHYARD_PORT=4001
~/.venv-py312/bin/python examples/orchestrator_advanced_demo.py
```

If you already launched the server in one terminal, run the demo in another terminal with `SWITCHYARD_PORT` set to the same port. The script now detects and reuses an existing server instead of trying to bind the port again.

To stop the manually started server, press `Ctrl+C` in the terminal running `switchyard serve`, or kill the port if it gets stuck:

```bash
lsof -iTCP:4001 -sTCP:LISTEN -t
kill <pid>
```

The demo now prompts you to select a task type and highlights routing decisions with `colorama`-powered output.

Sample tasks:

- Task 1: `Say hi and ask how the user is doing: 'Hey, how are you?'.`
  - showcases **Protocol Translation** and **Multi-Backend Routing** by using a chitchat prompt that should route to the lightweight chat profile.
- Task 2: `Create a compact 6-point onboarding checklist for a new developer joining a project.`
  - showcases **Strong Types** and **Profile-Owned Routing** by generating structured output through a more capable profile while preserving typed request/response wiring.
- Task 3: `Design a 5-step plan to launch a small Python service with CI/CD.`
  - showcases **Multi-Backend Routing** and **Request Statistics** by routing a more complex task through a stronger profile and printing per-turn latency and token usage.
- Task 4: Custom task: try `Write a concise API contract for a mobile app that reads sensor data and calls a cloud function.`
  - showcases **Profile-Owned Routing** and **signal-driven cascade** by allowing you to exercise custom routing behavior and fallback logic with a user-specific prompt.
- Task 5: Exit the demo.

These examples show how Switchyard can route different requests to different profiles/models based on the task type and classifier output.

What this demo shows:
- **Protocol Translation**: Switchyard converts between OpenAI Chat-compatible requests and the endpoint-specific request/response formats used by NVIDIA/Anthropic backends.
- **Multi-Backend Routing**: it uses classifier-driven routing plus a signal-driven cascade/failover path to choose between cheap and strong profiles.
- **Strong Types**: the underlying Switchyard engine uses typed request/response containers for OpenAI, Anthropic, and OpenAI Responses-style APIs.
- **Profile-Owned Routing**: each profile owns routing, backend calls, stats, and translation wiring, so the orchestrator can focus on high-level task selection.
- **One-Command Launchers**: while this demo uses `switchyard serve`, the same Switchyard platform also supports one-command launchers like `switchyard launch claude`, `switchyard launch codex`, and `switchyard launch openclaw` to spin up a local proxy and drop into the target CLI.
- **Request Statistics**: Switchyard can collect per-request latency, token, and cost data for routed requests.

Notes & next steps:
- You can replace the classifier with a local heuristic or an external signal
  (e.g., request headers, user metadata) to route requests differently.
- To demo cascade behavior thoroughly, you can adjust `FALLBACKS` in the
  script to change the order or add more profiles.

