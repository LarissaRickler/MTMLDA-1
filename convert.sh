#!/bin/sh

for file in *.dot
do
  dot -Tpng $file -o $file.png
done

ffmpeg -framerate 5 -i mtmc%05d.dot.png -vf scale=1280:720 -c:v libx264 -r 30 -pix_fmt yuv420p out.mp4
