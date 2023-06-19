# Quilt
[Quilt-1M: One Million Image-Text Pairs for Histopathology](https://quilt1m.github.io/)

![teaser](aux/quilt_main_img.jpeg "teaser")

### [Paper]() | [Huggingface Demo](https://huggingface.co/wisdomik/QuiltNet-B-32) 


## Abstract
>Recent accelerations in multi-modal applications have been made possible with the plethora of image and text data available online. However, the scarcity of similar data in the medical field, specifically in histopathology, has halted similar progress. To enable similar representation learning for histopathology, we turn to YouTube, an untapped resource of videos, offering k hours of valuable educational histopathology videos from expert clinicians. From YouTube, we curate Quilt: a large-scale vision-language dataset consisting of image and text pairs. Quilt was automatically curated using a mixture of models, including large language models, handcrafted algorithms, human knowledge databases, and automatic speech recognition. In comparison, the most comprehensive datasets curated for histopathology amass only around k samples. We combine Quilt with datasets, from other sources, including Twitter, research papers, and the internet in general, to create an even larger dataset: Quilt-1M, with M paired image-text samples, marking it as the largest vision-language histopathology dataset to date. We demonstrate the value of Quilt-1M by fine-tuning a pre-trained CLIP model. Our model outperforms state-of-the-art models on both zero-shot and linear probing tasks for classifying new pathology images across diverse patch-level datasets of different sub-pathologies and cross-modal retrieval tasks.
>
## News
- [x] *2023-03-03* Upated repository with links to models and data.
- [x] *2023-06-13* Inital ccode/data release .


## Requirements
```bash
conda create --name quilt python=3.9 && conda activate quilt
```
Then install [requirements/](data/requirements.txt)



## Pretrained Model
We provide the checkpoints for all QuiltNet finetuned models.
[ViT-B-32|GPT77/](https://huggingface.co/wisdomik/QuiltNet-B-32)
[ViT-B-16|GPT77/](https://huggingface.co/wisdomik/QuiltNet-B-32)
[ViT-B-16|PMB256/](https://huggingface.co/wisdomik/QuiltNet-B-16-PMB)


## Testing
Visualization of inputs and output:

![](aux/clip_heatmap.png)


## Citing Quilt-1M

```
@inproceedings{ikezogwo2023quilt,
  title={Quilt-1M: One Million Image-Text Pairs for Histopathology},
  author={Wisdom O. Ikezogwo, Mehmet S. Seyfioglu, Fatemeh Ghezloo, Dylan Geva , Fatwir S. Mohammed, Pavan K. Anand, Ranjay Krishna, Linda G. Shapiro.},
  booktitle={###},
  year={###}
}

```

## Acknowledgements

This code borrows heavily from and [open-clip](https://github.com/mlfoundations/open_clip) and [TiMM's library](https://github.com/huggingface/pytorch-image-models). We also thank the contributors of [merlot](https://github.com/rowanz/merlot).

## Maintenance

Please open a GitHub issue for any help. If you have any questions regarding the technical details, feel free to contact us.

## License
The codes and the pretrained model in this repository are under the MIT license as specified by the LICENSE file.
