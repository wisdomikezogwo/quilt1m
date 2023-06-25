
Each patch-level downstream histology dataset used in evalauation is properly cited in the preprint can downloaded either from their authors github, website or other data hosting services easily, and most datasets folders can be formatted to leverage torch's ImageFolder Dataset class for ease of use, we provide a few [custom dataset class](/eval/histopathology_datasets.py) for the datasets that arent easily parsed with ImageFolder .

For evaluation (zero-shot, retrieval, linear probing) we use [Clip benchmark](https://github.com/LAION-AI/CLIP_benchmark) with modified [classnames](/eval/en_classnames.json) for each dataset based off of its labels. The templates for zero-shot classification can be found in the paper. For linear probing, the only addition to [Clip benchmark](https://github.com/LAION-AI/CLIP_benchmark) was to run  hyperparameter search and multiple seed runs for all models and baselines, as reported in the paper.

