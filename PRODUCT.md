# Product

PortraitLocal is a local-first Windows application for conservative automatic
photo enhancement. Its product workflow is:

```text
Import -> Analyze -> Enhance -> Preview -> Export
```

## Release 0.4.0

The current release is a complete MVP with:

- a desktop UI and a batch CLI;
- JPG/JPEG/PNG/WebP input and new JPEG output;
- Natural, Balanced and Strong profiles;
- Mask R-CNN instance-aware person segmentation with an offline DeepLabV3
  fallback;
- YuNet face analysis and geometry-gated detail protection;
- explainable scene classification using existing detections and measurable
  image signals;
- conservative policies for portrait, group, landscape, city, architecture,
  indoor, night, food, animal and object scenes;
- adaptive non-generative denoise/sharpening;
- conditional GFPGAN face restoration and Real-ESRGAN 2x upscale;
- isolated output folders, processing logs, recipes and optional debug masks.

The heavy AI stages are conditional rather than applied to every image. The
ordinary enhancement path does not replace the original file and does not
silently perform generative editing.

## Explicit Boundaries

The 0.4.0 product does not include RAW development, photographic styles,
generative or edit modes, dedicated AI denoise/deblur/artifact models, queue
cancellation/retry, series consistency, automatic photo selection,
composition analysis or additional model families. Scene classification uses
bounded object and pixel signals rather than dense semantic segmentation.
