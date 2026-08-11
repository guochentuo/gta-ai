#!/usr/bin/env bash
set -euo pipefail

output_dir=${1:?output directory is required}
font=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf
bold=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf
mkdir -p "$output_dir/keyframes"

ffmpeg -hide_banner -loglevel error -y \
    -f lavfi -i color=c=0x0f172a:s=1200x628:d=1 \
    -vf "drawbox=x=0:y=0:w=1200:h=80:color=0x2563eb:t=fill,drawtext=fontfile=$bold:text='GTA ANALYTICS':fontcolor=white:fontsize=38:x=55:y=18,drawtext=fontfile=$bold:text='CUT REPORT TIME 30 PERCENT':fontcolor=white:fontsize=52:x=55:y=145,drawtext=fontfile=$font:text='14-DAY FREE TRIAL':fontcolor=0x93c5fd:fontsize=34:x=58:y=245,drawbox=x=55:y=335:w=300:h=80:color=0xf97316:t=fill,drawtext=fontfile=$bold:text='START NOW':fontcolor=white:fontsize=34:x=98:y=355,drawtext=fontfile=$font:text='Internal pilot with 12 analysts':fontcolor=0x94a3b8:fontsize=24:x=55:y=540" \
    -frames:v 1 "$output_dir/ad-creative.png"

ffmpeg -hide_banner -loglevel error -y \
    -f lavfi -i color=c=0xf8fafc:s=1280x720:d=1 \
    -vf "drawbox=x=0:y=0:w=1280:h=70:color=0x111827:t=fill,drawtext=fontfile=$bold:text='GTA DASHBOARD':fontcolor=white:fontsize=30:x=45:y=18,drawtext=fontfile=$bold:text='Traffic grew, conversions fell':fontcolor=0x111827:fontsize=48:x=55:y=115,drawbox=x=55:y=215:w=340:h=150:color=white:t=fill,drawtext=fontfile=$font:text='SESSIONS':fontcolor=0x64748b:fontsize=24:x=80:y=235,drawtext=fontfile=$bold:text='120,000':fontcolor=0x111827:fontsize=30:x=80:y=275,drawtext=fontfile=$bold:text='PLUS 20 PERCENT':fontcolor=0x15803d:fontsize=22:x=80:y=320,drawbox=x=455:y=215:w=340:h=150:color=white:t=fill,drawtext=fontfile=$font:text='CONVERSIONS':fontcolor=0x64748b:fontsize=24:x=480:y=235,drawtext=fontfile=$bold:text='3,600':fontcolor=0x111827:fontsize=30:x=480:y=275,drawtext=fontfile=$bold:text='MINUS 10 PERCENT':fontcolor=0xb91c1c:fontsize=22:x=480:y=320,drawbox=x=855:y=215:w=340:h=150:color=white:t=fill,drawtext=fontfile=$font:text='CVR':fontcolor=0x64748b:fontsize=24:x=880:y=235,drawtext=fontfile=$bold:text='3.0 PERCENT':fontcolor=0xb91c1c:fontsize=28:x=880:y=275,drawtext=fontfile=$font:text='WAS 4.0 PERCENT':fontcolor=0x64748b:fontsize=22:x=880:y=320,drawbox=x=55:y=420:w=1140:h=85:color=0xfef3c7:t=fill,drawtext=fontfile=$bold:text='TRACKING GAP  AUG 3  FROM 0200 TO 0400 UTC':fontcolor=0x92400e:fontsize=28:x=85:y=445,drawbox=x=55:y=565:w=330:h=75:color=0x2563eb:t=fill,drawtext=fontfile=$bold:text='REVIEW CAMPAIGN':fontcolor=white:fontsize=28:x=85:y=585" \
    -frames:v 1 "$output_dir/webpage-screenshot.png"

ffmpeg -hide_banner -loglevel error -y -f lavfi -i color=c=0x1d4ed8:s=640x360:d=1 \
    -vf "drawtext=fontfile=$bold:text='SCENE 1  ATLAS':fontcolor=white:fontsize=46:x=(w-text_w)/2:y=105,drawtext=fontfile=$font:text='Product introduction':fontcolor=white:fontsize=28:x=(w-text_w)/2:y=190" -frames:v 1 "$output_dir/keyframes/source-01.png"
ffmpeg -hide_banner -loglevel error -y -f lavfi -i color=c=0x15803d:s=640x360:d=1 \
    -vf "drawtext=fontfile=$bold:text='SCENE 2  BENEFIT':fontcolor=white:fontsize=42:x=(w-text_w)/2:y=105,drawtext=fontfile=$font:text='Save 30 minutes':fontcolor=white:fontsize=32:x=(w-text_w)/2:y=190" -frames:v 1 "$output_dir/keyframes/source-02.png"
ffmpeg -hide_banner -loglevel error -y -f lavfi -i color=c=0xc2410c:s=640x360:d=1 \
    -vf "drawtext=fontfile=$bold:text='SCENE 3  PRICE':fontcolor=white:fontsize=44:x=(w-text_w)/2:y=105,drawtext=fontfile=$font:text='29 USD per month':fontcolor=white:fontsize=32:x=(w-text_w)/2:y=190" -frames:v 1 "$output_dir/keyframes/source-03.png"
ffmpeg -hide_banner -loglevel error -y -f lavfi -i color=c=0xb91c1c:s=640x360:d=1 \
    -vf "drawtext=fontfile=$bold:text='SCENE 4  CTA':fontcolor=white:fontsize=46:x=(w-text_w)/2:y=105,drawtext=fontfile=$font:text='Start Free Trial':fontcolor=white:fontsize=32:x=(w-text_w)/2:y=190" -frames:v 1 "$output_dir/keyframes/source-04.png"

ffmpeg -hide_banner -loglevel error -y \
    -loop 1 -t 2 -i "$output_dir/keyframes/source-01.png" \
    -loop 1 -t 2 -i "$output_dir/keyframes/source-02.png" \
    -loop 1 -t 2 -i "$output_dir/keyframes/source-03.png" \
    -loop 1 -t 2 -i "$output_dir/keyframes/source-04.png" \
    -filter_complex "[0:v][1:v][2:v][3:v]concat=n=4:v=1:a=0,format=yuv420p[v]" \
    -map "[v]" -r 30 -c:v libx264 "$output_dir/video-fixture.mp4"

ffmpeg -hide_banner -loglevel error -y -i "$output_dir/video-fixture.mp4" \
    -vf "fps=1/2" -q:v 2 "$output_dir/keyframes/frame-%02d.jpg"
