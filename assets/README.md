# Showcase asset

`workflow.gif` was generated from the
[Sao Paulo M3DC aerial LiDAR tile MDS color 2222 212](https://laz-m3dc-pmsp.s3.sa-east-1.amazonaws.com/MDS_color_2222-212.laz).
The source archive is published by the Prefeitura Municipal de Sao Paulo and
is listed under the
[GNU General Public License version 3](https://registry.opendata.aws/pmsp-lidar/).

The 1600 by 900 animation contains a deterministic sample of 1,600,000 points
from the urban part of the tile. Building and tree labels were produced by
`facebook/sam3` through the `segment-lidar` prompt workflow. Run
`scripts/make_animation.py` to reproduce the asset.
