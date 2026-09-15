# Providers

Replaceable model implementations belong here, including segmentation, face
detection, restoration, denoise and upscale providers.

Every provider must expose its model identity and version for processing recipes.
The primary person provider preserves per-instance confidence values before
merging masks for photographic correction; the semantic provider is an offline
fallback, not a second concurrent pipeline.
