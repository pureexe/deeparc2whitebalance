
import argparse, os, cv2
import numpy as np 
import matplotlib.pyplot as plt
from multiprocessing import Pool

def read_exr(image_path):
    image = cv2.imread(image_path,  cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)  
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image = np.clip(image,0,1) # remove bad pixel above 1 
    image = np.power(image,1) # gamma correction
    return image

def read_mask(mask_path):
    mask = cv2.imread(mask_path)
    if len(mask.shape) == 3 or mask.shape[2] == 3:
        mask = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
    return mask > 0

def normalize_pixel(image,mask):
    average_pixel = np.average(image[mask],axis=0) 
    image = image / average_pixel
    image = np.clip(image,0,1) # image should be in [0,1]
    return image

def get_masks(mask_dir, num_thread = 8):
    mask_files = os.listdir(mask_dir)
    mask_paths = []
    masks = {}
    mask_name_list = []
    for mask_file in mask_files:
        if not mask_file.endswith('.png'):
            continue
        mask_paths.append(os.path.join(mask_dir,mask_file))
        mask_name_list.append(mask_file.split('.')[0])
    with Pool(num_thread) as p:
        mask_list = p.map(read_mask,mask_paths)
    for i in range(len(mask_list)):
        masks[mask_name_list[i]] = mask_list[i]
    return masks 

def apply_mask(packed_data):
    image = read_exr(packed_data['image_path'])
    image = normalize_pixel(image,packed_data['mask'])
    plt.imsave(packed_data['output_path'],image)

def apply_masks(input_dir, output_dir, masks, num_thread = 8):
    packed_datas = []
    exr_files = os.listdir(input_dir)
    for exr_file in exr_files:
        if not exr_file.endswith('.exr'):
             continue
        packed_datas.append({
            'image_path': os.path.join(input_dir,exr_file),
            'mask': masks[exr_file.split('_')[0]],
            'output_path': os.path.join(output_dir, exr_file.split('.')[0] + '.png')
        })
    with Pool(num_thread) as p:
        p.map(apply_mask,packed_datas)

def main(args):
    masks = get_masks(args.reference, args.thread)
    apply_masks(args.input, args.output, masks, args.thread)
 
def entry_point():
    parser = argparse.ArgumentParser(
        description='deeparc2whitebalance.py - white balance the camera that output from deeparc')
    parser.add_argument(
        '-i',
        '--input',
        type=str,
        default='D:/Datasets/yellow/',
        help='input exr file directory',
    )
    parser.add_argument(
        '-r',
        '--reference',
        type=str,
        default='mask/',
        help='mask to use as reference'
    ) 
    parser.add_argument(
        '-o',
        '--output',
        type=str,
        default='output/',
        help='output directory')     
    parser.add_argument(
        '--thread',
        type=int,
        default=8,
        help='output directory')     
    main(parser.parse_args())


if __name__ == "__main__":
    entry_point()