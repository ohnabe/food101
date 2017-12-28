#!/usr/bin bash

nvidia-docker run --rm -it -v $(pwd):/work -v /srv/data/food101/food-101:/data -w /work actcast_cl:chainer3.2.0_git \
python3 /work/extract_rgb.py
