# python preprocess_VIRAT/video_annotation.py
# python preprocess_VIRAT/video_annotation_old.py

python preprocess_VIRAT/crop_videos.py  -o data/videos/val/cropped_128/ -s 128 -c 1 -a val
python preprocess_VIRAT/sort_cropped.py -i data/videos/val/cropped_128/ -o data/videos/val/cropped_128_sorted/
python preprocess_VIRAT/crop_videos.py  -o data/videos/train/cropped_128/ -s 128 -c 1 -a train
python preprocess_VIRAT/sort_cropped.py -i data/videos/train/cropped_128/ -o data/videos/train/cropped_128_sorted/

# python preprocess_VIRAT/crop_videos.py  -o data/videos/with_context/cropped_128/ -s 128 -c 2
# python preprocess_VIRAT/sort_cropped.py -i data/videos/with_context/cropped_128/ -o data/videos/with_context/cropped_128_sorted/

# python preprocess_VIRAT/visualize.py     -o data/videos/vis/
# python preprocess_VIRAT/visualize_old.py -o data/videos/vis_old/
