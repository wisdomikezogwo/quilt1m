# quilt1m
Repository for Quilt dataset


# quilt
[Quilt-1M: One Million Image-Text Pairs for Histopathology](https://quilt1m.github.io/)

QUILT-1M is an image-text dataset for histopathology. Curated from educational videos on Youtube QUILT-1M contributes the largest dataset for vision language modeling in histopathology.

Visit our project page at [quilt1m.github.io](https://quilt1m.github.io/) to learn more.

![teaser](aux/quilt_main_img.jpeg "teaser")

## What's here

We are releasing the following:
* Code for generating images and alinging with released text descriptions in [data/](data/)
* Models trained on Quilt-1M and models used to curate Quilt-1M

We plan to release:
* Code for pretraining CLIP models on Quilt-1M
* Evaluation (zero-shot, linear probing etc. ) code as well.

## Enviroment and setup



```bash
conda create --name quilt python=3.9 && conda activate quilt
```
Then install [requirements/](data/requirements.txt)



### Bibtex
```
@inproceedings{ikezogwo2023quilt,
  title={Quilt-1M: One Million Image-Text Pairs for Histopathology},
  author={Wisdom O. Ikezogwo, Mehmet S. Seyfioglu, Fatemeh Ghezloo, Dylan Geva , Fatwir S. Mohammed, Pavan K. Anand, Ranjay Krishna, Linda G. Shapiro.},
  booktitle={###},
  year={###}
}

```
