@echo off
rem One-click setup + launch for Windows (needs Python 3.12 from python.org, "py" launcher)
cd /d "%~dp0"
if not exist .venv (py -3.12 -m venv .venv || py -m venv .venv)
call .venv\Scripts\activate
python -m pip install -q -r requirements.txt
streamlit run app.py
