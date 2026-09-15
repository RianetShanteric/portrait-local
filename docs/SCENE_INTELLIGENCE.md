# Scene Intelligence 0.4.0

Portrait Local classifies a photograph locally before rendering. The result is
not a generative caption and is not sent anywhere. It combines object labels
already produced by the installed Mask R-CNN model with measurable image
signals: sky colour in the upper frame, vegetation colour, structural lines and
low-light distribution.

## Scene Contract

The primary classes are `portrait`, `group`, `landscape`, `city`,
`architecture`, `indoor`, `night`, `food`, `animal`, `object` and `general`.
Night and indoor can also be secondary tags. A tiny pedestrian in a street
scene does not force the photograph into portrait mode: people take priority
only when a face is visible or the combined person mask occupies a meaningful
part of the frame.

Every recipe records:

- primary scene, confidence and secondary tags;
- bounded scores for all scene candidates;
- short reasons based on the strongest signals;
- detected object label, confidence and relative area.

Object bounding boxes and facial landmark coordinates are transient. They are
not written to recipes.

## Conservative Policies

- night preserves the low-key intent, increases denoise slightly and reduces
  sharpening and background correction;
- city and architecture retain structural detail while limiting denoise;
- landscape and food reduce saturation/background intervention;
- portraits and groups keep the established identity-protected path;
- weak evidence falls back to `general` instead of inventing certainty.

All inference is offline after installation. No image, crop, metadata or recipe
is uploaded by the application.
