from pathlib import Path
from pydantic import computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


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
    output_csv: Path = Path("./output/results.csv")

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
