# Robit

Robit is a small, locally operated home robot with a browser control panel,
voice conversation, on-device AI, camera perception, and person tracking. Its
motors run on an ESP32 controller while a Windows PC runs the voice, vision,
and control services.

For technical details, see [docs/architecture.md](docs/architecture.md) and
[DESIGN.md](DESIGN.md).

## Setup

Install 64-bit Python 3.13, then open PowerShell in this folder and run:

```powershell
py -3.13 --version
.\Scripts\setup.bat
```

## Start

```powershell
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
.\Scripts\run.bat
```

When Robit is ready, open [http://localhost:8080](http://localhost:8080) to
use the control panel.

## Stop

In the PowerShell window running Robit, press `Ctrl+C`.
