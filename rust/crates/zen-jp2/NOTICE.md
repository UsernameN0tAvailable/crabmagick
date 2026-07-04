# zen-jp2 — Attribution

This crate is a fork of [`hayro-jpeg2000`](https://github.com/LaurenzV/hayro/tree/main/hayro-jpeg2000)
v0.4.0 by Laurenz Stampfl, licensed Apache-2.0 OR MIT.

Changes from the upstream:
- Crate renamed to `zen-jp2`
- Added `rayon`-based parallel tile decode (disjoint-write unsafe scatter)
- Integrated into the IIIF worker via `zen_jp2::api::decode`
- Removed `moxcms` / hayro `image`-crate integration
- Added simple `api::DecodedImage` public API

The original crate is part of the Hayro PDF renderer:
https://github.com/LaurenzV/hayro

License: Apache-2.0 OR MIT (unchanged from upstream).
