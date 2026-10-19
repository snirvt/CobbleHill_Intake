import logging
import platform
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, SecretStr, computed_field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from _SECRETS._secrets import API_TOKEN, CLIENT_ID, CLIENT_SECRET, GRAPH_DRIVE_ID, TENANT_ID

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

logger = logging.getLogger(__name__)


def is_dgx_spark() -> bool:
    """True on the DGX Spark prod box (the only aarch64 Linux machine we run on).

    Apple Silicon reports ``arm64``, not ``aarch64``, so a Mac is treated as a laptop.
    """
    return platform.machine().lower() == "aarch64"


class PageTargetRule(BaseModel):
    """Extract only the page holding ``section_header`` for a matching note.

    A note whose parsed (role, category) equals this rule's has its content
    narrowed to the single page containing ``section_header`` instead of the
    whole document. ``page_hint`` (1-indexed) is tried first; if the header is
    not on it, all pages are scanned. Notes that never contain the header fall
    back to full text.
    """

    role: str
    category: str | None
    section_header: str
    page_hint: int  # 1-indexed page most likely to hold the section


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="COBBLEHILL_")

    base_path: Path = _PROJECT_ROOT
    data_folder: Path = Path("data")
    node_bin_paths: list[str] = [
        "/home/snir/.nvm/versions/node/v22.22.2/bin",  # nvm (this machine)
        str(Path.home() / ".nvm/versions/node/v22.22.2/bin"),
        "/usr/local/bin",   # Homebrew / standard Linux
        "/usr/bin",         # system node
        "/opt/homebrew/bin",  # Mac M1/M2
    ]
    supported_extensions: dict[str, str] = {
        ".pdf": "pdf",
        ".txt": "txt",
        ".jpg": "image",
        ".jpeg": "image",
        ".png": "image",
    }
    max_concurrent_files: int = 10
    # Role prefix a note filename must start with (after an optional category prefix).
    note_role_prefixes: dict[str, str] = {"dr": "dr_", "nurse": "nurse_"}
    # Optional category prefixes that may precede the role prefix, e.g. hospital_nurse_x.pdf.
    # A note without any of these has category None. Pairing happens within one category.
    note_categories: list[str] = ["hospital", "peds"]
    # Notes matching (role, category) are narrowed to a single section page
    # instead of their full (50+ page) text. Add a rule to target more sections.
    page_target_rules: list[PageTargetRule] = [
        PageTargetRule(
            role="nurse",
            category="hospital",
            section_header="Admission/ROC Summary",
            page_hint=48,
        ),
    ]
    # Categories the pair diagnosis check runs on. Other categories are skipped
    # entirely (no LLM call, no issue flagged).
    diagnosis_check_categories: list[str] = ["hospital"]
    # Categories the pair care-needed check runs on (request or medical necessity).
    care_check_categories: list[str] = ["peds"]
    # Max words in a main-diagnosis grouping label; longer LLM labels are truncated.
    main_diagnosis_max_words: int = 4
    # Dr-validity check: patients younger than this (years) need a diagnosis AND medical
    # necessity; older patients need only a diagnosis.
    pediatric_age_threshold: float = 18
    ocr_dpi: int = 300  # DPI liteparse renders PDF pages at before OCR; images use easyocr, not this.
    # easyocr (image OCR). Reads images directly, no PDF conversion.
    ocr_languages: list[str] = ["en"]
    # easyocr GPU per machine. ocr_gpu is auto-filled from these (Spark vs laptop)
    # unless set explicitly, e.g. COBBLEHILL_OCR_GPU=false.
    ocr_gpu_laptop: bool = False
    ocr_gpu_spark: bool = True
    ocr_gpu: bool = False  # placeholder; _apply_machine_profile fills it
    # Custom weights dir; pair with ocr_download=False for offline/air-gapped use.
    ocr_model_dir: str | None = None
    # Let easyocr fetch model weights on first run (weights only, no patient data).
    ocr_download: bool = True
    ollama_url: str = "http://localhost:11434"
    # Which server runs the LLM. "ollama" (default) is the single-user laptop path;
    # "vllm" is the on-prem DGX Spark path. Opt in with COBBLEHILL_LLM_BACKEND=vllm.
    llm_backend: Literal["ollama", "vllm"] = "ollama"
    ollama_model: str = "gemma3:1b"
    # ollama_model: str = "medgemma:27b"
    # vLLM: OpenAI-compatible server shared with NurseBookSkill on the Spark. Must be
    # on-prem (loopback / private address) — the provider refuses public hosts.
    vllm_base_url: str = "http://localhost:8001"
    vllm_model: str = "nemotron-3.5-lightning"  # the server's --served-model-name
    # Sampling per the Nemotron model card.
    vllm_temperature: float = 0.6
    vllm_top_p: float = 0.95
    # Extra request params sent to vLLM per thinking level. Levels are data: tune or add
    # one here, no code change. "thinking_token_budget" needs vLLM support for it; an
    # unsupported key is ignored by the server (check out_tok in the timing logs).
    vllm_thinking_levels: dict[str, dict[str, Any]] = {
        "off": {"chat_template_kwargs": {"enable_thinking": False}},
        "low": {
            "chat_template_kwargs": {"enable_thinking": True},
            "thinking_token_budget": 1024,
        },
        "high": {"chat_template_kwargs": {"enable_thinking": True}},  # unlimited
    }
    # Thinking level per LLM task (keys match the timing log step names), used by the
    # dr-validity and dr-nurse-match pipelines. Ollama ignores it.
    # Override e.g. COBBLEHILL_TASK_THINKING='{"diagnosis": "low", ...}'.
    task_thinking: dict[str, str] = {
        "age": "off",
        "diagnosis": "off",
        "care": "off",
        "main_diagnosis": "off",
        "clinical_match": "off",
    }
    classifier_tasks: list[str] = ["doctor_visit_needed"]
    max_concurrent_llm_calls: int = 8
    # Attempts per LLM step (call + parse) before the step is reported as failed.
    llm_max_attempts: int = 3
    # Initial backoff (seconds) between LLM attempts; doubles each retry. 0 = no wait.
    llm_retry_wait_s: float = 1.0
    # Hard cap on tokens generated per LLM call (thinking + answer). Stops a model stuck in a
    # repetition loop from generating until the context window is full (seen: 260k tokens, 80 min).
    llm_max_output_tokens: int = 8192
    # Wall-clock limit per LLM call; a timed-out call fails and is retried like any other error.
    llm_request_timeout_s: float = 300
    sharepoint_download_concurrency: int = 5
    # Subfolder of the SharePoint input folder that results are uploaded into.
    sharepoint_results_subfolder: str = "results"
    # Upload target for local runs, which have no SharePoint input folder to inherit.
    sharepoint_results_folder: str = "Patient Encounters/Medical Notes/results"
    output_path: Path = Path("output/results.xlsx")
    sharepoint_client_id: str = CLIENT_ID
    sharepoint_tenant_id: str = TENANT_ID
    sharepoint_client_secret: str = CLIENT_SECRET
    sharepoint_drive_id: str = GRAPH_DRIVE_ID
    # FastAPI server (classifier/api.py). Default localhost; override with --host / --port
    # or COBBLEHILL_API_HOST / COBBLEHILL_API_PORT.
    api_host: str = "127.0.0.1"
    api_port: int = 8010
    # Shared secret clients send as the X-API-Key header. Comes from _SECRETS API_TOKEN
    # (never committed); COBBLEHILL_API_KEY overrides it. Required beyond localhost.
    api_key: SecretStr | None = SecretStr(API_TOKEN)

    @model_validator(mode="before")
    @classmethod
    def _apply_machine_profile(cls, data: Any) -> Any:
        """Fill ocr_gpu for this machine unless explicitly provided."""
        if not isinstance(data, dict):
            return data
        profile = "spark" if is_dgx_spark() else "laptop"
        if "ocr_gpu" not in data:
            source = f"ocr_gpu_{profile}"
            data["ocr_gpu"] = data.get(source, cls.model_fields[source].default)
        logger.info("Machine profile %s: ocr_gpu=%s", profile, data["ocr_gpu"])
        return data

    @model_validator(mode="after")
    def _check_task_thinking(self) -> "Settings":
        unknown = {
            task: level
            for task, level in self.task_thinking.items()
            if level not in self.vllm_thinking_levels
        }
        if unknown:
            raise ValueError(
                f"task_thinking uses unknown levels {unknown}; "
                f"known: {sorted(self.vllm_thinking_levels)}"
            )
        return self

    @model_validator(mode="after")
    def _resolve_relative_paths(self) -> "Settings":
        if not self.data_folder.is_absolute():
            self.data_folder = self.base_path / self.data_folder
        if not self.output_path.is_absolute():
            self.output_path = self.base_path / self.output_path
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def node_bin_path(self) -> str:
        """Return first path in node_bin_paths that contains a node executable."""
        for p in self.node_bin_paths:
            if (Path(p) / "node").exists():
                return p
        raise ValueError(
            f"No node binary found. Searched: {self.node_bin_paths}. "
            "Add your node bin dir to COBBLEHILL_NODE_BIN_PATHS."
        )


settings = Settings()
