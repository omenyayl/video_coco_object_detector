#!/bin/bash

videos_dir=$1

SAVEIFS=$IFS
IFS=$(echo -en "\n\b")

if [ ! -d $video_dir ]
then
	echo "Usage: $0 <path to videos>"
	exit 1
fi

declare -a videos_to_process=('/media/hblab/Seagate Expansion Drive/*.MOV')

script_dir="$( cd -P "$( dirname "$0" )" && pwd )"
process_video_script_dir='./detect_video.py'
cd ${script_dir}


process_video () {
    video="$1"
    video_dir="$(dirname "$video")"
    video_no_path="${video##*/}"
	video_name="${video_no_path%.*}"

    mkdir "$video_dir/$video_name"
    python ${process_video_script_dir} \
        --model ./models/bdd100k_16.h5 \
        --video "${video}" \
        --output "$video_dir/$video_name" \
        --n-frames 10
}


for video in $videos_dir/*
do
	echo "Processing $video"
    process_video "${video}"
done
