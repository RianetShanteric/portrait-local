# Regression Dataset

Regression assets must have a documented source and a license that permits local
storage, modification and redistribution with the repository. Do not add random
search-result images.

## Dataset Manifest

`dataset.json` is the canonical inventory. It records the source page, author,
license, original download hash, repository hash, privacy transformation,
coverage goals and stable expectations for every case. All stored images have
zero EXIF fields; the CC0 back-view source contained precise GPS coordinates,
which are deliberately absent from the repository derivative.

## Cases

### `couple_outdoors`

`images/couple_outdoors_public_domain.jpg`

- Source: Wikimedia Commons, “Couple outdoors.jpg” (National Cancer Institute
  Visuals Online, image 4500)
- Author: Bill Branson (photographer); National Cancer Institute (source), an
  agency of the U.S. federal government
- License: public domain (U.S. federal government work)
- Stored size: 3000 x 1993 JPEG
- Coverage: two faces, mixed skin tones, fine hair, dark/light clothing and a
  bright detailed background

The source image remains byte-for-byte unchanged.

### `back_view_seated`

`images/back_view_seated_cc0.jpg`

- Source: Wikimedia Commons, “Back view of a seated person”
- Author: OutdoorShooter00
- License: CC0 1.0 Universal
- Coverage: person segmentation without a visible face, indoor mixed light and
  the no-face restoration gate
- Privacy: the source contained timestamp and precise GPS coordinates; the
  stored 2250 x 3000 derivative contains no EXIF metadata

### `indoor_moderator`

`images/indoor_moderator_public_domain.jpg`

- Source: Wikimedia Commons / National Cancer Institute, “Health Professional
  Addresses Group”
- Author: Bill Branson
- License: public domain (U.S. federal government work)
- Coverage: foreground occlusion, background people, mixed indoor colour and a
  prominent face
- Privacy: the stored 1955 x 3000 derivative contains no EXIF metadata

### `longfellow_group`

`images/longfellow_group_public_domain.jpg`

- Source: Wikimedia Commons / National Cancer Institute, “Longfellow Group”
- Author: National Cancer Institute
- License: public domain (U.S. federal government work)
- Coverage: ten-person monochrome group, small faces, low detail and the
  conditional upscale gate
- Privacy: the stored 700 x 464 derivative contains no EXIF metadata; its size
  intentionally exercises the conditional upscale gate

### `glacier_landscape`

`images/glacier_landscape_public_domain.jpg`

- Source: Wikimedia Commons / National Park Service, “National Park Service
  (48754075203)”
- Author: NPS Natural Resources
- License: public domain (U.S. federal government work)
- Coverage: a people-free landscape with sky, forest and water reflection
- Privacy: the stored 2200 x 1467 derivative contains no EXIF metadata

### `city_night`

`images/city_night_public_domain.jpg`

- Source: Wikimedia Commons / National Cancer Institute, “City street at night”
- Author: Chris Spielmann
- License: public domain (U.S. federal government work)
- Coverage: night city, traffic lights, vehicles and tiny background pedestrians
- Privacy: the stored 2200 x 1453 derivative contains no EXIF metadata

## Baseline Evidence

Each case has an auditable folder under `baseline/balanced` containing source
and output hashes, analysis JSON, person/skin masks and a visual comparison.

To rebuild them from an installed local checkout:

```powershell
.\.venv\Scripts\python.exe scripts\build_regression_baselines.py
```

The builder verifies every input hash before processing. A baseline change must
be reviewed visually; hashes alone are insufficient evidence of a quality
change.

## Coverage Gaps

Still missing: extreme backlight, animals, food, isolated architecture, motion
blur and severe compression artifacts.
