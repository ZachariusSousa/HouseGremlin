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
.\setup
```

## Start

```powershell
.\run
```

When Robit is ready, open [http://localhost:8080](http://localhost:8080) to
use the control panel.

## Stop

Press `Ctrl+C` in the PowerShell window running Robit for a clean shutdown.
If an earlier stack was interrupted, run this from another PowerShell window:

```powershell
.\stop
```
