# Data Preparation
To Reconstruct Quilt:

- Download videos using video-ids in the [data csv files](https://drive.google.com/drive/folders/1B-CiMD6UfMxEY3t8xckOc3tK1FBI0DOm?usp=sharing) to the directory `${BASE_DIR}/videos/...` with subfolders as described below.

- Download ensemble of histopathology classifiers:

-[histo_cyto](https://huggingface.co/wisdomik/QuiltNet-B-32/blob/main/cyto_histo.pth.tar)
-[all_inclusive](https://huggingface.co/wisdomik/QuiltNet-B-32/blob/main/all_inclusive.pth.tar)
-[dino_checkpoint](https://huggingface.co/wisdomik/QuiltNet-B-32/blob/main/dino_checkpoint.pth.tar)
to the directory `${BASE_DIR}/histo_models/...` as described below.

- To generate images and pair them with text run.
    ```python 
    BASE_DIR="/path/to/project"
    python -m main --base_dir ${BASE_DIR}
    ```

The data structure is like this:
```
dataset
├── /path/to/project
│  ├── video
│  │  ├── channel_id
│  │  │     ├── video_id_name
│  │  │            ├──video_id_name.mp4
│  │  │     ...

│  │  │     ├── video_id_name
│  │  │            ├──video_id_name.mp4
│  ├── frames
│  │  ├── channel_id
│  │  │     ├── video_id_name
│  │  │            ├──video_id_name.mp4

│  │  │     ...

│  │  │     ├── video_id_name
│  │  │            ├──video_id_name.mp4
│  ├── histo_models
│  │  ├── cyto_histo.pth.tar
│  │  ├── all_inclusive.pth.tar
│  │  ├── dino_checkpoint.pth.tar
│  ├── quilt_recon.csv
│  ├── quilt_data.csv

```

* Please note, there may be some alignment issues as differences in stack and mostly videos or frames extracted can effect image alignment.