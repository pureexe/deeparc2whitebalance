
import argparse
import os 
import numpy as np 
import matplotlib.pyplot as plt
import cv2
 
def gamma_correction(image,gamma = 1):
    return np.power(image,1)

def remove_bad_pixel(image):
    return np.clip(image,0,1)

def read_exr(image_path):
    image = cv2.imread(image_path,  cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)  
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image = remove_bad_pixel(image)
    image = gamma_correction(image)
    return image

def read_mask(mask_path):
    mask = cv2.imread(mask_path)
    if len(mask.shape) == 3 or mask.shape[2] == 3:
        mask = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
    return mask > 0

def avg_pixel(image,mask):
    return np.average(image[mask],axis=0)

def normalize_pixel(image,avg_px):
    image = image / avg_px
    image = np.clip(image,0,1)
    return image

def get_masks(mask_dir, thread_number = 8):
    mask_files = os.listdir(mask_dir)
    masks = {}
    for mask_file in mask_files:
        if not mask_file.endswith('.png'):
            continue
        mask_name = mask_file.split('.')[0]
        mask_pixel = read_mask(os.path.join(mask_dir,mask_file))
        masks[mask_name] = mask_pixel
    return masks 


def apply_masks(input_dir, output_dir, masks):
    exr_files = os.listdir(input_dir)
    for exr_file in exr_files:
        if not exr_file.endswith('.exr'):
            continue
        mask = masks[exr_file.split('_')[0]]
        image = read_exr(os.path.join(input_dir,exr_file))
        average_pixel = avg_pixel(image,mask)
        image = normalize_pixel(image,average_pixel)
        file_name = exr_file.split('.')[0]
        plt.imsave(os.path.join(output_dir,file_name + '.png'),image)

def main(args):
    masks = get_masks(args.reference, args.thread)
    apply_masks(args.input, args.output, masks)

    #image = read_exr(os.path.join(args.input,'cam000_00000.exr'))
    #mask = read_mask(os.path.join(args.reference,'cam000_00000.png'))
    #average_pixel = avg_pixel(image,mask)
    #image = normalize_pixel(image,average_pixel)
    #plt.imshow(image)
    #plt.show()
 
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