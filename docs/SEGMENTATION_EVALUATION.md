# Person Segmentation

## Current Implementation

PortraitLocal 0.4.0 uses Torchvision Mask R-CNN ResNet-50 FPN (COCO V1) as the
primary person provider. Torchvision DeepLabV3 ResNet-50 remains an offline
fallback when the instance model is not installed.

Both implementations run locally through the existing provider contract.
Torchvision code is BSD-3-Clause; pretrained-data provenance remains documented
separately in the model manifest. No Ultralytics or AGPL component is used.

## RTX 5070 Evaluation

Warm local inference, one image at a time, using the pinned Torch 2.7.0 /
Torchvision 0.22.0 environment. Times are diagnostic measurements, not fixed
performance promises.

| Case | DeepLab instances | Mask R-CNN instances | DeepLab seconds | Mask R-CNN seconds |
| --- | ---: | ---: | ---: | ---: |
| couple_outdoors | 1 merged class mask | 2 | 0.65 | 0.73 |
| back_view_seated | 1 merged class mask | 1 | 0.35 | 0.68 |
| indoor_moderator | 1 merged class mask | 1 | 0.27 | 0.65 |
| longfellow_group | 1 merged class mask | 10 | 0.15 | 0.89 |

Mask R-CNN is modestly slower, but it turns the formerly inferred people count
into actual detected instances. It also detects the back-facing person without
depending on a face. The masks were visually reviewed on all four licensed
regression images.

## Runtime Policy

- maximum analysis side is 1280 pixels;
- only COCO class `person` is accepted;
- instance score must be at least 0.68;
- tiny masks below 0.035% of the inference frame are discarded;
- accepted instances are merged only after their individual confidences have
  been retained for counting and audit recipes;
- missing primary weights select the local DeepLab fallback;
- inference never downloads weights.
