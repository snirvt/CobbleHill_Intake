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
- Full install / troubleshooting / NGC-container fallback: NurseBookSkill
  `agent/VLLM.md` and `agent/deploy/README.md`.
