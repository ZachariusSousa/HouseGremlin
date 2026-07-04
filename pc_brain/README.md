# PC Brain

This service runs on your computer and owns the expensive work: camera streaming, LLM/tool calling, speech, autonomy, and logs.

## Run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:ROBIT_BASE_URL="http://192.168.1.50"
uvicorn app.main:app --reload --host 0.0.0.0 --port 8080
```

Replace `192.168.1.50` with the IP printed by the robot firmware.

## Endpoints

- `GET /health`
- `GET /robot/status`
- `POST /robot/drive`
- `POST /robot/head`
- `POST /robot/stop`
- `GET /tools`

The LLM/tool layer should call these PC endpoints, not the ESP firmware directly.
