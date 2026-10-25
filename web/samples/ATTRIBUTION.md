# Real-photo fixture attribution

These photographs are separate from the repository's MIT-licensed source code.
No downloads are made during tests. Photos are public-domain/CC0 as documented by the original scikit-image data maintainers.

| Source | Author | License | Evidence |
| --- | --- | --- | --- |
| [astronaut](https://flic.kr/p/r9qvLn) | NASA | Public domain | [scikit-image documentation](https://scikit-image.org/docs/0.25.x/api/skimage.data.html#skimage.data.astronaut) |
| [chelsea](https://scikit-image.org/docs/0.25.x/api/skimage.data.html#skimage.data.chelsea) | Stefan van der Walt | CC0-1.0 | [scikit-image documentation](https://scikit-image.org/docs/0.25.x/api/skimage.data.html#skimage.data.chelsea) |
| [coffee](https://scikit-image.org/docs/0.25.x/api/skimage.data.html#skimage.data.coffee) | Rachel Michetti | CC0-1.0 | [scikit-image documentation](https://scikit-image.org/docs/0.25.x/api/skimage.data.html#skimage.data.coffee) |
| [rocket](https://www.flickr.com/photos/spacexphotos/16511594820/) | SpaceX | Public domain | [scikit-image documentation](https://scikit-image.org/docs/0.25.x/api/skimage.data.html#skimage.data.rocket) |
| [grass](https://www.deviantart.com/linolafett/art/Grass-01-434853879) | linolafett | CC0-1.0 | [scikit-image documentation](https://scikit-image.org/docs/0.25.x/api/skimage.data.html#skimage.data.grass) |
| [clock](https://scikit-image.org/docs/0.25.x/api/skimage.data.html#skimage.data.clock) | Stefan van der Walt | Public domain | [scikit-image documentation](https://scikit-image.org/docs/0.25.x/api/skimage.data.html#skimage.data.clock) |

The manifest records original source hashes, actual fixture hashes, decoded-pixel hashes, dimensions and derivation steps. Alpha, portrait, wide and thumbnail variants are explicitly derived from the six photos. Test masks/boxes are geometric probes and are not semantic annotations.

Rebuild only when intentionally updating fixtures: install scikit-image 0.25.2 and Pillow 12.3.0, then run `python tools/prepare_real_images.py --source-dir <skimage/data>`.
