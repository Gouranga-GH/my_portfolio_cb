import os
from pathlib import Path


def main():
    # Simple entry-point that forwards to the Streamlit app
    project_root = Path(__file__).resolve().parent
    target = project_root / "apps" / "gj_rag_chat" / "main.py"
    if not target.exists():
        raise SystemExit("Streamlit app not found at apps/gj_rag_chat/main.py")
    # Launch via streamlit if run directly
    os.system(f"streamlit run {str(target)}")


if __name__ == "__main__":
    main()


