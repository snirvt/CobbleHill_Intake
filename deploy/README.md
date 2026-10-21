# Deploy: vLLM on the DGX Spark

`vllm.service` is a copy of NurseBookSkill's `agent/deploy/vllm.service`. It runs the
**one shared** vLLM server (Nemotron-3.5-Lightning-30B-A3B NVFP4, served as
`nemotron-3.5-lightning` on `127.0.0.1:8001`) that both NurseBookSkill and CobbleHill
Intake use. Same unit name in both repos: install it from either, never run two.
Keep the two copies identical when changing flags.

## Point CobbleHill at it
```bash
echo COBBLEHILL_LLM_BACKEND=vllm >> .env      # default backend is ollama
```
Matching settings (`config/settings.py`): `vllm_base_url=http://localhost:8001`,
`vllm_model=nemotron-3.5-lightning`.

## Install (only if not already installed by NurseBookSkill)
Unit assumes box user `cobblehill` and server venv `~/vllm-venv` — edit
`User`/`Group`/`WorkingDirectory`/`ExecStart` if different.
```bash
systemctl status vllm --no-pager || {
  sudo cp deploy/vllm.service /etc/systemd/system/
  sudo systemctl daemon-reload
  sudo systemctl enable --now vllm.service
}
```

## Verify
```bash
journalctl -u vllm -f                                                  # model load takes minutes
curl -sS http://localhost:8001/v1/models | jq -r '.data[0].id'         # nemotron-3.5-lightning
```

## Notes
- `--reasoning-parser nemotron_v3` is required: it keeps the model's thinking out of
  the reply `content`, which CobbleHill's classifiers parse as JSON.
- `--gpu-memory-utilization 0.45` leaves room for other GPU users (e.g. easyocr with
  `ocr_gpu` on). Restarting `vllm` interrupts NurseBookSkill too.
- `--max-num-seqs 16`: max requests decoded together across both apps. Decode is
  memory-bound, so batching raises total tok/s almost for free. CobbleHill sends up to
  `max_concurrent_llm_calls` (8) at once.
- Thinking per task: `task_thinking` / `vllm_thinking_levels` in `config/settings.py`
  (default `off`). Sent per request, so NurseBookSkill keeps its own thinking.
- Full install / troubleshooting / NGC-container fallback: NurseBookSkill
  `agent/VLLM.md` and `agent/deploy/README.md`.

# Deploy: CobbleHill API service (auto-restart on `git pull`)

`cobblehill-api.service` runs the API as a systemd **user** service: it starts on boot,
restarts if it crashes, and the git hooks in `git-hooks/` restart it after every `git pull`
that changes code. No sudo needed for day-to-day use.

## Install once (on Spark, as `cobblehill`, from the repo folder)
Stop any API you started by hand first (Ctrl+C), or port 8010 will be taken.
```bash
sudo loginctl enable-linger cobblehill      # FIRST: starts your user systemd; keeps it on boot
# Point systemctl --user at it (VS Code terminals can carry stale values). Add these two
# lines to ~/.bashrc so every new terminal has them.
export XDG_RUNTIME_DIR=/run/user/$(id -u)
export DBUS_SESSION_BUS_ADDRESS=unix:path=$XDG_RUNTIME_DIR/bus
systemctl --user status                     # must print a tree, not an error
mkdir -p ~/.config/systemd/user
cp deploy/cobblehill-api.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now cobblehill-api
git config core.hooksPath deploy/git-hooks  # restart the API on every git pull
curl localhost:8010/health                  # check it answers
```
The unit expects the repo at `~/CobbleHill_Intake`; edit `WorkingDirectory` if it is elsewhere.

**"Failed to connect to bus: Connection refused"**: your user systemd is not running or the
terminal points at the wrong place. Run the `enable-linger` and the two `export` lines above,
then retry. Still failing → check `loginctl show-user cobblehill` (`Linger=yes`, `State=`)
and `ls -la /run/user/$(id -u)/` (should contain `bus`).

## Day to day
```bash
git pull                                       # restarts the API if code changed
systemctl --user status cobblehill-api         # running?
journalctl --user -u cobblehill-api -f         # logs
systemctl --user restart cobblehill-api        # manual restart
```
A restart cancels requests that are running at that moment. If `uv.lock` or
`pyproject.toml` changed, the hook runs `uv sync` before restarting.
