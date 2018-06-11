#!/bin/bash

declare -a videos_to_process=('./data/123.MOV')

script_dir="$( cd -P "$( dirname "$0" )" && pwd )"
process_video_script_dir='./detect_video.py'
cd ${script_dir}


process_video () {
    video=$1
    video_dir=$(dirname "$video")
    video_no_path="${video##*/}"
	video_name="${video_no_path%.*}"

    mkdir "$video_dir/$video_name"
    python ${process_video_script_dir} \
        --model /home/oleg/models/retina.h5 \
        --video ${video} \
        --output "$video_dir/$video_name" \
        --n-frames 10
}


for video in "${videos_to_process[@]}"
do
    process_video ${video}
done


