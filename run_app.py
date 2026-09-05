import streamlit.web.cli as stcli
import sys

if __name__ == "__main__":
    sys.argv = [
        "streamlit",
        "run",
        "app.py",
        "--server.address=127.0.0.1",
        "--server.port=8501",
        "--server.headless=true"
    ]
    stcli.main()