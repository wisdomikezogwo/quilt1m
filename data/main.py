import os
import ast
import cv2
import glob
import copy
import torch
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from skimage.metrics import structural_similarity
from model_utils import (get_model_ensemble, single_clf_inference, single_vit_inference)
from data_utils import (get_histo_srt_im_recon, save_frame_chunks_recon)



# Construct the argument parser
parser = argparse.ArgumentParser('Reconstruct Quilt')

parser.add_argument('--base_dir', default='./quilt', #'convnext_tiny_384_in22ft1k',#
                    type=str, help='base directory for data')

parser.add_argument('--data_csv', default='data_df.csv',
                    type=str, help='path to per video_id reconstruction data in csv file')

parser.add_argument('--recon_csv', default='recon_df.csv',
                    type=str, help='path to per image-(text) pair csv file')

parser.add_argument('--network', default='convnext_tiny.fb_in22k_ft_in1k_384', #'convnext_tiny_384_in22ft1k',#
                    type=str, help='name of pretrained network to use')
parser.add_argument('--histo_classes', default=2, type=int,
                    help='histo_classes')
parser.add_argument('-is', '--input_size', default=384, type=int,
                    help='image size for resizing')
parser.add_argument('--crop_size', default=300, type=int,
                    help='crop_size to extract from center of image before resizing and passing to the model')
args = parser.parse_args([])


def main(args, data_df, recon_df, device, histo_models_dict, video_paths_dict):
    df_vid_ids = list(set(recon_df['video_id'].tolist()))
    
    for vid_id in df_vid_ids:
        vid_recon_df = recon_df[recon_df['video_id'] == vid_id]
        times = ast.literal_eval(vid_recon_df.iloc[0]["key_frame_times"])
        predictions = ast.literal_eval(vid_recon_df.iloc[0]["predictions"])
        list_idle_im_se_t_tup = ast.literal_eval(vid_recon_df.iloc[0]["stable_times"])
        video_path = video_paths_dict[vid_id]
        
        temp_df = data_df[data_df['video_id'] == vid_id]
        pair_chunk_time = float(temp_df.iloc[0]['pair_chunk_time'])
        height = temp_df.iloc[0]['height']
        width = temp_df.iloc[0]['width']
        fps = temp_df.iloc[0]['fps']
        
        chunks, chunks_images, chunks_image_times = get_histo_srt_im_recon(times, predictions, pair_chunk_time)
        
        rep_chunks_image_paths = []  
        for chunk_id, chunk in enumerate(chunks):
            rep_chunk_im = []    
            stable_times = list_idle_im_se_t_tup[chunk_id][chunk_id][0]

            rep_chunk_im_temp = save_frame_chunks_recon(video_path, stable_times, chunk_id,fps, height, width)

            # cleanup with histo classifier
            for idx , img in enumerate(rep_chunk_im_temp):
                if img is None:
                    continue

                img_pil = Image.fromarray(cv2.cvtColor(np.copy(img), cv2.COLOR_BGR2RGB))
                pred_histo = 0
                for name, histo_model in histo_models_dict.items():
                    if name != 'dino' and not pred_histo:
                        pred_histo += single_clf_inference(histo_model, img_pil, input_size=args.input_size,
                                                        crop_size=args.crop_size, device=device)
                    elif name == 'dino' and not pred_histo:
                        model, linear_classifier = histo_model
                        pred_histo += single_vit_inference(model, linear_classifier, img_pil,
                                                            input_size=384, crop_size=224, device=device)

                if pred_histo:
                    rep_chunk_im.append(img)
                else:
                    pass
            
            # no stable regions, use chunks_image_times, deduplicate with ssim
            if not rep_chunk_im: 
                video = cv2.VideoCapture(video_path)
                chunk_im_ts = chunks_image_times[chunk_id]#index_

                video.set(cv2.CAP_PROP_POS_MSEC, chunk_im_ts[0] * 1000)
                _, frame1 = video.read()

                rep_chunk_im.append(np.copy(frame1))

                resized_frame1 = cv2.resize(frame1, (512, 512))
                h, w = resized_frame1.shape[:2]
                crop_size = 64

                for ts in chunk_im_ts[1:]:
                    video.set(cv2.CAP_PROP_POS_MSEC, ts * 1000)
                    _, frame2 = video.read()

                    resized_frame2 = cv2.resize(frame2, (512, 512))
                    top_left_x = np.random.randint(int(w / 2) - crop_size)
                    top_left_y = np.random.randint(int(h / 2) - crop_size)
                    crop1 = resized_frame1[top_left_y: top_left_y + crop_size,
                            top_left_x: top_left_x + crop_size, :]

                    crop2 = resized_frame2[top_left_y: top_left_y + crop_size,
                            top_left_x: top_left_x + crop_size, :]
                    ssim_val = structural_similarity(crop1, crop2, multichannel=True)

                    if ssim_val < 0.02:
                        resized_frame1 = cv2.resize(copy.deepcopy(np.copy(frame2)), (512, 512))
                        rep_chunk_im.append(frame2)

            rep_chunks_image_paths.append(rep_chunk_im)
        
        name_image_dict = {}
        for chunk_id, _ in enumerate(chunks):
            df_chunk_image_names = temp_df[temp_df['chunk_id'] == chunk_id]['image_path'].tolist()
            rep_images = rep_chunks_image_paths[chunk_id]
            for name_idx, names in enumerate(df_chunk_image_names):
                try:
                    name_image_dict[names] = rep_images[name_idx]
                except:
                    pass
        
        for img_name, img_object in name_image_dict.items():
            if type(img_object) != Image.Image:
                img_object = Image.fromarray(cv2.cvtColor(np.copy(img_object), cv2.COLOR_BGR2RGB))
            im_path = os.path.join(FRAMES_DIR, img_name)
            img_object.save(im_path) 





if __name__ == '__main__':
    args = parser.parse_args()

    BASE_DIR = args.base_dir
    VIDEO_DIR = os.path.join(BASE_DIR, "videos")
    FRAMES_DIR = os.path.join(BASE_DIR, "frames")
    args.model_dir = MODELS_DIR = os.path.join(BASE_DIR, "histo_models") 

    if not os.path.exists(FRAMES_DIR):
        Path(FRAMES_DIR).mkdir(parents=True, exist_ok=True)

    video_paths = glob.glob(os.path.join(VIDEO_DIR, '*', '*.mp4'))
    video_paths_dict = {os.path.splitext(os.path.basename(video_path))[0]: video_path for video_path in video_paths}


    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # load model for histo classification of frames
    histo_models_names = [f'{MODELS_DIR}/cyto_histo.pth.tar',
                        f'{MODELS_DIR}/all_inclusive.pth.tar']
    histo_models_dict = get_model_ensemble(args, histo_models_names, device)

    recon_df = pd.read_csv(os.path.join(BASE_DIR, 'recon_df.csv')
    data_df = pd.read_csv(os.path.join(BASE_DIR, 'data_df.csv'))

    main(args, data_df, recon_df, device, histo_models_dict, video_paths_dict)

    # After all images have been saved, clean csv of all image-only rows
    # post process to only extract rows of data_df with 
    data_df = data_df[data_df['it_pair']==True]
    data_df['medical_text'] = data_df['medical_text'].apply(ast.literal_eval)
    data_df = data_df.explode('medical_text')
    data_df.to_csv(os.path.join(BASE_DIR, 'data_df.csv'))

    # ToDo: add code to remove all rows with no image found.

    