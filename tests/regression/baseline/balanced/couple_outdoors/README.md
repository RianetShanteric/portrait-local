# Balanced 0.4.0 Baseline

This baseline was produced locally on NVIDIA GeForce RTX 5070 with the pinned
environment in `requirements.lock`.

- Input SHA-256: `67F77E9689FE1D204443045C9E62D5B9969299DA5CF9F36016E45541CD2DADA7`
- Output SHA-256: `5C79F7EFBC0B332BDDE38F8EDCA96EE262BF6A61438E5226E4FDDE32C6BD4344`
- Detected people: 2 independent Mask R-CNN instances
- Detected faces: 2
- Face restoration: skipped for both faces
- Landmark-qualified faces: 2
- Eye/hair detail: conservative luminance-only pass
- Teeth: detected and protected, not whitened
- Denoise: skipped
- Sharpen: skipped
- Upscale: skipped

The image was already clean and sharp. The accepted change is a masked subject
lift with mild background highlight/saturation control and protected skin tones.
`recipe.json` is the canonical machine-readable evidence.
