# Portrait Detail Policy

Portrait regions are derived locally from the five landmarks already returned
by YuNet. No additional model, network call or generative face edit is involved.

## Safety Gates

A face is eligible only when:

- YuNet confidence is at least 0.82;
- the shorter face side is at least 96 pixels;
- all five landmarks remain near the face box;
- eye and mouth distances are plausible for the detected face;
- both eyes, nose and mouth have a valid vertical order.

Failed geometry produces no eye, teeth or hair operation. This deliberately
skips tiny faces, strong poses and uncertain detections.

## Regions

- Eyes: two soft ellipses based on the eye landmarks. Only low-amplitude
  luminance detail is added; eye shape and colour are never changed.
- Teeth: bright, low-saturation pixels inside the landmark-derived mouth region.
  They are protected from adaptive denoise/sharpen and never whitened.
- Hair: the non-skin part of a head prior, intersected with the detected person
  mask and excluding eyes and mouth. Only low-amplitude luminance detail is
  added.

Natural, Balanced and Strong scale the same bounded detail policy. Every run
records eligible-face count, mask area, applied amounts and teeth protection in
its recipe. Debug output includes all three masks and a landmark overlay.
Raw landmark coordinates are not written to recipes; they exist only in memory
during the local processing run.
