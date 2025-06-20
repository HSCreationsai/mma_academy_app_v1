import os
from huggingface_hub import snapshot_download
import toml


def ensure_model_directory(path):
    """Create the model directory if it doesn't exist"""
    os.makedirs(path, exist_ok=True)


def download_deepseek():
    # Load path from secrets.toml
    try:
        secrets = toml.load(".streamlit/secrets.toml")
        model_path = secrets["llm"]["model_path"]

        print(f"⬇️ Downloading DeepSeek-7B to {model_path} ...")
        ensure_model_directory(model_path)

        snapshot_download(
            repo_id="deepseek-ai/deepseek-llm-7b-base",
            local_dir=model_path,
            local_dir_use_symlinks=False,
            resume_download=True
        )
        print(f"✅ DeepSeek-7B downloaded successfully to {model_path}")

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("Make sure you're in the correct directory and have proper permissions.")


if __name__ == "__main__":
    download_deepseek()