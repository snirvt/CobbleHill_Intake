from pathlib import Path
from pydantic import computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict
from _SECRETS._secrets import CLIENT_ID, TENANT_ID, CLIENT_SECRET, GRAPH_DRIVE_ID


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="COBBLEHILL_")

    data_folder: Path = Path("./data")
    node_bin_paths: list[str] = [
        "/home/snir/.nvm/versions/node/v22.22.2/bin",  # nvm (this machine)
        str(Path.home() / ".nvm/versions/node/v22.22.2/bin"),
        "/usr/local/bin",   # Homebrew / standard Linux
        "/usr/bin",         # system node
        "/opt/homebrew/bin",  # Mac M1/M2
    ]
    supported_extensions: dict[str, str] = {".pdf": "pdf", ".txt": "txt"}
    max_concurrent_files: int = 10
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "gemma3:1b"
    classifier_tasks: list[str] = ["doctor_visit_needed"]
    max_concurrent_llm_calls: int = 3
    sharepoint_download_concurrency: int = 5
    sharepoint_results_folder: str = "Patient Encounters/Medical Notes/Non-Admits-results"
    output_path: Path = Path("./output/results.xlsx")
    sharepoint_client_id: str = CLIENT_ID
    sharepoint_tenant_id: str = TENANT_ID
    sharepoint_client_secret: str = CLIENT_SECRET
    sharepoint_drive_id: str = GRAPH_DRIVE_ID

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
