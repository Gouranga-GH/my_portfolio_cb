"""Streamlit entrypoint for Cloud.

On Streamlit Cloud, the `main` file is executed by Streamlit directly.
We should import our app's `main()` and call it, not spawn a new
`streamlit run` process.
"""

from apps.gj_rag_chat.main import main  # noqa: E402


if __name__ == "__main__":
    main()


