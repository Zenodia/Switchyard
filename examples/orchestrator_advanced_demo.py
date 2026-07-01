#!/usr/bin/env python3
"""Advanced orchestrator demo showing dynamic routing and cascade/failover.

Behavior:
- Uses `examples/nvidia_profile.yaml` and the local Switchyard server.
- Classifies an incoming task to decide which agent/profile to use (dynamic routing).
- Demonstrates a cascade/failover pattern: if the chosen agent fails or returns
  an unsatisfactory result, the orchestrator retries with a secondary agent.

Flow:
1. Start `switchyard serve --config examples/nvidia_profile.yaml`.
2. For a given `user_task`, ask a lightweight classifier model (`nano`) to
   pick among: `chitchat`, `heavy`, `thinker`.
3. Route the request to the selected primary profile.
4. If the response is empty, too short, contains an error, or the request
   times out, fallback to a secondary profile (cascade).
5. Aggregate and print the final result and a short routing trace.

Run:
- Ensure env keys are set and `switchyard` is in PATH or set `SWITCHYARD_CLI`.
- `python examples/orchestrator_advanced_demo.py`

"""
import asyncio
import json
import os
import socket
import subprocess
import sys
import time
from typing import Optional, Tuple

import httpx
from colorama import Fore, Style, init

init(autoreset=True)

VENV_SWITCHYARD = os.environ.get("SWITCHYARD_CLI", "/home/ubuntu/.venv-py312/bin/switchyard")
CONFIG_YAML = "examples/nvidia_profile.yaml"
SWITCHYARD_PORT = int(os.environ.get("SWITCHYARD_PORT", "4001"))
BASE_URL = f"http://127.0.0.1:{SWITCHYARD_PORT}"

# Mapping of logical agent names to profile ids in the YAML
PROFILES = {
    "chitchat": "nano",
    "heavy": "nano-30b",
    "thinker": "ultra",
}

# Fallback order per logical agent (cascade)
FALLBACKS = {
    "chitchat": ["nano", "gpt-4o-mini-profile"],
    "heavy": ["nano-30b", "deepseek"],
    "thinker": ["ultra", "nano-30b"],
}

STRONG_TYPES_PROFILES = {"nano-30b", "deepseek"}

FEATURES_BY_TASK = {
    "1": [
        ("Protocol Translation", "This task routes a lightweight chitchat prompt through translated request/response handling."),
        ("Multi-Backend Routing", "The orchestrator chooses a primary profile and may fall back on alternate profiles."),
    ],
    "2": [
        ("Strong Types", "This task uses stronger typed profile routing for structured request/response handling."),
        ("Profile-Owned Routing", "The selected profile owns routing, backend calls, and typed translation wiring."),
    ],
    "3": [
        ("Multi-Backend Routing", "A higher-complexity task routes to a stronger profile and may cascade to a fallback."),
        ("Request Statistics", "The demo prints per-turn latency and usage information for this task."),
    ],
    "4": [
        ("Profile-Owned Routing", "Custom prompts are routed through profile-aware decision logic."),
        ("Cascade/Failover", "The orchestrator can fall back to alternate profiles if the first response is unsatisfactory."),
    ],
}

TASK_OPTIONS = {
    "1": (
        "chitchat",
        "Say hi and ask how the user is doing: 'Hey, how are you?'.",
    ),
    "2": (
        "heavy",
        "Create a compact 6-point onboarding checklist for a new developer joining a project.",
    ),
    "3": (
        "thinker",
        "Design a 5-step plan to launch a small Python service with CI/CD.",
    ),
    "4": (
        "custom",
        "Enter your own task prompt.",
    ),
    "5": (
        "exit",
        "Exit the multi-turn demo.",
    ),
}


def select_user_task() -> Tuple[str, str]:
    print(f"{Fore.CYAN}Select the type of task to route through Switchyard:{Style.RESET_ALL}")
    for key, (_, prompt) in TASK_OPTIONS.items():
        print(f"  {Fore.YELLOW}{key}{Style.RESET_ALL}. {prompt}")

    choice = input(f"{Fore.GREEN}Choose a task [1-5]: {Style.RESET_ALL}").strip()
    if choice == "4":
        custom = input(f"{Fore.GREEN}Enter your custom task prompt:{Style.RESET_ALL} ").strip()
        if custom:
            return choice, custom
        print(f"{Fore.RED}No custom task entered; using default thinker prompt.{Style.RESET_ALL}")
        return "3", TASK_OPTIONS["3"][1]
    if choice == "5":
        return choice, "__exit__"

    if choice in TASK_OPTIONS:
        return choice, TASK_OPTIONS[choice][1]

    print(f"{Fore.RED}Invalid choice; using default thinker task.{Style.RESET_ALL}")
    return "3", TASK_OPTIONS["3"][1]


def pretty_route(title: str, message: str) -> None:
    print(f"{Fore.BLUE}{title}{Style.RESET_ALL} {message}")


def start_switchyard_server() -> subprocess.Popen:
    cmd = [VENV_SWITCHYARD, "serve", "--config", CONFIG_YAML, "--port", str(SWITCHYARD_PORT)]
    print("Starting switchyard server:", " ".join(cmd))
    proc = subprocess.Popen(cmd)
    return proc


def wait_for_server(port: int, timeout: int = 15) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.settimeout(1.0)
                sock.connect(("127.0.0.1", port))
                return True
            except Exception:
                time.sleep(0.5)
    return False


async def post_chat(model: str, prompt: str, timeout: int = 60) -> Tuple[bool, Optional[str], float, Optional[dict]]:
    """Post a chat completion to the local Switchyard server.
    Returns (ok, text_or_error, duration_seconds, raw_response).
    """
    url = f"{BASE_URL}/v1/chat/completions"
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}]}
    start = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(url, json=payload)
            duration = time.perf_counter() - start
            resp.raise_for_status()
            data = resp.json()
            # Extract OpenAI Chat shape safely
            try:
                text = data["choices"][0]["message"]["content"]
                return True, text, duration, data
            except Exception:
                return True, json.dumps(data), duration, data
    except Exception as e:
        duration = time.perf_counter() - start
        return False, str(e), duration, None


async def classify_task(task: str) -> str:
    """Use a lightweight profile to classify the task into logical agent.
    Returns one of the keys in PROFILES.
    """
    classifier_prompt = (
        "Classify the following user request into exactly one of: chitchat, heavy, thinker."
        " Respond with a single word (chitchat|heavy|thinker) and nothing else.\n\n"
        f"Request: {task}"
    )
    ok, resp, _, _ = await post_chat(PROFILES["chitchat"], classifier_prompt, timeout=20)
    if not ok or not resp:
        # conservative default
        return "thinker"
    token = resp.strip().lower().split()[0]
    if token in PROFILES:
        return token
    # fallback heuristic
    if "code" in task or "script" in task or "deploy" in task:
        return "heavy"
    return "thinker"


def is_unsatisfactory(text: Optional[str]) -> bool:
    if not text:
        return True
    if len(text.strip()) < 20:
        return True
    lower = (text or "").lower()
    if any(k in lower for k in ["error", "could not", "timeout", "failed"]):
        return True
    return False


def format_usage(data: Optional[dict]) -> str:
    if not data:
        return ""
    usage = data.get("usage") or data.get("prompt_usage") or {}
    if not isinstance(usage, dict):
        return ""
    fields = []
    for key in ["prompt_tokens", "completion_tokens", "total_tokens", "estimated_cost"]:
        if key in usage:
            fields.append(f"{key}={usage[key]}")
    return ", ".join(fields)


async def route_with_cascade(logical_agent: str, prompt: str) -> Tuple[str, str, float, Optional[dict]]:
    """Try primary profile then cascade through fallbacks until satisfactory response.
    Returns (profile_used, response_text, duration_seconds, raw_response).
    """
    tried = []
    print(f"{Fore.MAGENTA}Routing through profiles for logical agent:{Style.RESET_ALL} {logical_agent}")
    for profile in FALLBACKS.get(logical_agent, FALLBACKS.get(logical_agent, [])):
        tried.append(profile)
        pretty_route("Trying profile:", profile)
        ok, resp, duration, data = await post_chat(profile, prompt, timeout=60)
        summary = resp or "<no response>"
        pretty_route("Response length:", str(len(summary)))
        pretty_route("Latency:", f"{duration*1000:.0f} ms")
        usage_line = format_usage(data)
        if usage_line:
            pretty_route("Usage:", usage_line)
        if ok and not is_unsatisfactory(resp):
            pretty_route("Selected profile:", profile)
            return profile, resp, duration, data
        pretty_route("Fallbacking from:", profile)
    # if nothing good, return last try result (ok flag ignored)
    return tried[-1] if tried else "", resp or "", duration, data


async def get_routing_stats() -> Optional[dict]:
    url = f"{BASE_URL}/v1/routing/stats"
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            return resp.json()
    except Exception:
        return None


async def advanced_orchestrate(task_choice: str, user_task: str):
    trace = []
    # 1. classify
    chosen = await classify_task(user_task)
    trace.append(("classifier", chosen))

    # define three subtasks for demonstration
    chitchat_prompt = f"Friendly 2-line greeting about: {user_task}"
    heavy_prompt = f"Give a compact 6-point actionable checklist for: {user_task}"
    thinker_prompt = (
        f"You are an expert planner. Create a 5-step setup script for: {user_task}"
    )

    # pick prompt based on classification
    prompt_map = {"chitchat": chitchat_prompt, "heavy": heavy_prompt, "thinker": thinker_prompt}
    primary_prompt = prompt_map[chosen]

    profile_used, response, duration, data = await route_with_cascade(chosen, primary_prompt)
    trace.append(("executed_profile", profile_used))
    trace.append(("response_summary", (len(response or ""), (response or "")[:200].replace("\n", " "))))
    trace.append(("roundtrip_ms", round(duration * 1000)))
    usage_line = format_usage(data)
    if usage_line:
        trace.append(("usage", usage_line))

    print("\n--- ADVANCED ORCHESTRATOR OUTPUT ---\n")
    print("User task:\n", user_task)
    print("\nRouting trace:\n")
    for t in trace:
        print(" -", t)

    feature_lines = FEATURES_BY_TASK.get(task_choice, [])
    if feature_lines:
        print("\nFeature highlights:\n")
        for label, desc in feature_lines:
            pretty_route(Fore.CYAN + label + ":" + Style.RESET_ALL, desc)

    if profile_used in STRONG_TYPES_PROFILES:
        pretty_route(Fore.CYAN + "Strong Types enabled:" + Style.RESET_ALL,
                     "Routed through a typed profile with structured request/response handling.")

    print("\nRequest stats for this turn:\n")
    pretty_route("Profile:", profile_used)
    pretty_route("Latency:", f"{round(duration * 1000)} ms")
    if usage_line:
        pretty_route("Usage:", usage_line)
    else:
        pretty_route("Usage:", "none available")

    stats = await get_routing_stats()
    if stats:
        print("\nSwitchyard routing stats:\n")
        pretty_route("Total requests:", str(stats.get("total_requests", "n/a")))
        pretty_route("Average latency:", str(stats.get("average_latency_ms", "n/a")))
        pretty_route("Total tokens:", str(stats.get("total_tokens", "n/a")))
        pretty_route("Requests by profile:", json.dumps(stats.get("requests_by_profile", {}), indent=2))
    else:
        print("\nSwitchyard routing stats endpoint unavailable or empty. Per-turn latency/usage is shown above.\n")

    print("Final response (truncated):\n", (response or "")[:1000])
    print("\n------------------------------------\n")


async def test_failover_simulation():
    """Demonstrates explicit failure handling by calling an invalid profile first.
    This simulates a backend failing and then falling back to a working profile.
    """
    print("\n--- FAILOVER SIMULATION ---\n")
    # deliberately call a non-existent profile to force error
    ok, resp, _, _ = await post_chat("nonexistent-profile", "Say hi.", timeout=10)
    if not ok:
        print("Primary failed as expected:", resp)
        # fallback to a real profile
        ok2, resp2, _, _ = await post_chat(PROFILES["chitchat"], "Say hi.", timeout=10)
        print("Fallback response:\n", resp2)
    else:
        print("Unexpected success on primary (environment may map unknown ids differently).\nResp:\n", resp)


def main():
    if not os.environ.get("NVIDIA_API_KEY") or not os.environ.get("INFERENCE_API_KEY"):
        print("Warning: NVIDIA_API_KEY and/or INFERENCE_API_KEY not set in environment.")

    # If SWITCHYARD_PORT is already bound, assume a server is already running on that port.
    if not wait_for_server(SWITCHYARD_PORT, timeout=1):
        proc = start_switchyard_server()
        try:
            print("Waiting for server to start...")
            if not wait_for_server(SWITCHYARD_PORT, timeout=20):
                print("Server did not start in time; check logs.")
                proc.kill()
                sys.exit(1)
        except Exception:
            proc.kill()
            raise
    else:
        proc = None
        print(f"Detected existing Switchyard server on port {SWITCHYARD_PORT}; reusing it.")

    try:
        while True:
            task_choice, user_task = select_user_task()
            if user_task == "__exit__":
                print(f"{Fore.CYAN}Exiting the multi-turn demo.{Style.RESET_ALL}")
                break
            asyncio.run(advanced_orchestrate(task_choice, user_task))
            print(f"\n{Fore.GREEN}Turn complete. You can select another task or exit.{Style.RESET_ALL}\n")

        # Run failover demo after the interactive session
        asyncio.run(test_failover_simulation())

    finally:
        if proc is not None:
            print("Stopping switchyard server started by this script...")
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except Exception:
                proc.kill()


if __name__ == "__main__":
    main()
