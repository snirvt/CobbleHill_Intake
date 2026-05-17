from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="COBBLEHILL_")

    data_folder: Path = Path("./data")
    node_bin_path: str = "/home/snir/.nvm/versions/node/v22.22.2/bin"
    supported_extensions: dict[str, str] = {".pdf": "pdf"}
    max_concurrent_files: int = 10
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "gemma3:1b"
    classifier_tasks: list[str] = ["doctor_visit_needed"]
    max_concurrent_llm_calls: int = 3
    output_csv: Path = Path("./output/results.csv")


settings = Settings()
