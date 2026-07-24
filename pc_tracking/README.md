# Robit RF-DETR sidecar

This local-only service keeps RF-DETR and its Transformers 5 dependency out of
the voice environment. It accepts shared broker JPEGs and returns only `person`
detections. It does not retain images.

The default `ROBIT_TRACKING_DEVICE=auto` selects CUDA/BF16 when supported and
otherwise runs on CPU. Model-load failures leave the service available for
health checks while `/detect` returns a clear `503`.

To force CPU for one run from PowerShell:

```powershell
$env:ROBIT_TRACKING_DEVICE = "cpu"
.\Scripts\run.bat
```
