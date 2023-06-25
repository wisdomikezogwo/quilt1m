import os
import sys
import json
import glob
import PIL
import warnings
from pathlib import Path
import numpy as np
import torch
from torchvision import datasets
from torchvision.transforms import Compose

from subprocess import call
from collections import defaultdict
from torch.utils.data import default_collate
from PIL import Image
import pandas as pd
from torchvision.datasets import ImageFolder, PCAM


class MhistDataset(torch.utils.data.Dataset):
    def __init__(self, root, csv_file, image_dir, transform=None, train=True):
        csv_file = os.path.join(root, csv_file)
        image_dir = os.path.join(root, image_dir)

        self.data = pd.read_csv(csv_file)
        if train:
            self.data = self.data[self.data['Partition'] == 'train']
        else:
            self.data = self.data[self.data['Partition'] != 'train']
        self.image_paths = self.data['Image Name'].values
        self.labels = self.data['Majority Vote Label'].values
        self.image_dir = image_dir
        self.transform = transform
        self.train = train
        self.cat_to_num_map = {'HP': 0, 'SSA': 1}
        self.classes = ["hyperplastic polyp", "sessile serrated adenoma"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir, self.image_paths[index])
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        label = self.cat_to_num_map[self.labels[index]]

        return image, label


class SicapDataset(torch.utils.data.Dataset):
    def __init__(self, root, image_dir, transform=None, train=True):

        image_dir = os.path.join(root, image_dir)

        if train:
            csv_file = os.path.join(root, "partition/Test", "Train.xlsx")
            self.data = pd.read_excel(csv_file)
        else:
            csv_file = os.path.join(root, "partition/Test", "Test.xlsx")
            self.data = pd.read_excel(csv_file)

        # drop all columns except image_name and the label columns
        label_columns = ['NC', 'G3', 'G4', 'G5']  # , 'G4C']
        self.data = self.data[['image_name'] + label_columns]

        # get the index of the maximum label value for each row
        self.data['labels'] = self.data[label_columns].idxmax(axis=1)

        # replace the label column values with categorical values
        self.cat_to_num_map = label_map = {'NC': 0, 'G3': 1, 'G4': 2, 'G5': 3}  # , 'G4C': 4}
        self.data['labels'] = self.data['labels'].map(label_map)

        self.image_paths = self.data['image_name'].values
        self.labels = self.data['labels'].values
        self.image_dir = image_dir
        self.transform = transform
        self.train = train
        self.classes = ["non-cancerous well-differentiated glands",
                        "gleason grade 3 with atrophic well differentiated and dense glandular regions",
                        "gleason grade 4 with cribriform, ill-formed, large-fused and papillary glandular patterns",
                        "gleason grade 5 with nests of cells without lumen formation, isolated cells and pseudo-roseting patterns",
                        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir, self.image_paths[index])
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        label = self.labels[index]

        return image, label


class ArchCsvDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, transforms, img_key='image_path', caption_key='caption', sep=","):
        df = pd.read_csv(csv_file, sep=sep)
        self.images = df[img_key].tolist()
        self.captions = df[caption_key].tolist()
        self.transforms = transforms
        self.ids = list(sorted(df['ids'].tolist()))


    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        id_ = self.ids[idx]
        images = self.transforms(Image.open(str(self.images[id_])))
        texts = [str(self.captions[id_])]
        return images, texts


class OsteoDataset(torch.utils.data.Dataset):
    def __init__(self, root, csv_file, image_dir, transform=None):
        csv_file = os.path.join(root, csv_file)
        image_dir = os.path.join(root, image_dir)

        self.data = pd.read_csv(csv_file)
        self.data = self.data[self.data['classification'] != "viable: non-viable"]
        
        self.image_paths = self.data['image_name'].values
        self.labels = self.data['classification'].values
        self.image_dir = image_dir
        self.transform = transform
        self.cat_to_num_map = {'Non-Tumor': 0, 'Non-Viable-Tumor': 1, 'Viable': 2}
        self.classes = ["non-tumor", "non-viable necrotic osteosarcoma tumor", "viable osteosarcoma tumor"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir, self.image_paths[index])
        image_path = image_path.replace(' - ', '-')
        image_path = glob.glob(f"{image_path.replace(' ', '-')}*")[0]
        image = Image.open(image_path.replace(' ', '-')).convert('RGB')

        if self.transform:
            image = self.transform(image)

        label = self.cat_to_num_map[self.labels[index]]

        return image, label


class SkinDataset(torch.utils.data.Dataset):
    def __init__(self, root, csv_file, transform=None, train=True, val=False,
                 tumor=False):
        csv_file = os.path.join(root, csv_file)
        self.data = pd.read_csv(csv_file)

        if train:
            self.data = self.data[self.data['set'] == 'Train']
        else:
            if val:
                self.data = self.data[self.data['set'] == "Validation"]
            else:
                self.data = self.data[self.data['set'] == 'Test']

        if tumor:
            self.data = self.data[self.data['malignicy'] == 'tumor']
        self.tumor = tumor

        self.image_paths = self.data['file'].values
        self.labels = self.data['class'].values

        self.transform = transform
        self.train = train

        self.cat_to_num_map = {'nontumor_skin_necrosis_necrosis': 0,
                               'nontumor_skin_muscle_skeletal': 1,
                               'nontumor_skin_sweatglands_sweatglands': 2,
                               'nontumor_skin_vessel_vessel': 3,
                               'nontumor_skin_elastosis_elastosis': 4,
                               'nontumor_skin_chondraltissue_chondraltissue': 5,
                               'nontumor_skin_hairfollicle_hairfollicle': 6,
                               'nontumor_skin_epidermis_epidermis': 7,
                               'nontumor_skin_nerves_nerves': 8,
                               'nontumor_skin_subcutis_subcutis': 9,
                               'nontumor_skin_dermis_dermis': 10,
                               'nontumor_skin_sebaceousglands_sebaceousglands': 11,
                               'tumor_skin_epithelial_sqcc': 12,
                               'tumor_skin_melanoma_melanoma': 13,
                               'tumor_skin_epithelial_bcc': 14,
                               'tumor_skin_naevus_naevus': 15
                               }

        self.tumor_map = {'tumor_skin_epithelial_sqcc': 0,
                          'tumor_skin_melanoma_melanoma': 1,
                          'tumor_skin_epithelial_bcc': 2,
                          'tumor_skin_naevus_naevus': 3
                          }

        self.classes = list(self.cat_to_num_map)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        if not self.tumor:
            label = self.cat_to_num_map[self.labels[index]]
        else:
            label = self.tumor_map[self.labels[index]]

        return image, label

