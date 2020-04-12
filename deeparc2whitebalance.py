
import argparse, os, cv2
import numpy as np 
import matplotlib.pyplot as plt
from multiprocessing import Pool
from timeit import default_timer as timer

def remove_bad_pixel(image, percentile = None):
    if percentile is not None:
        _, _,c = image.shape
        threshold_pixel = np.percentile(image.reshape(-1,c),percentile,axis=0)
        for i in  range(c):
            image[:,:,i] = np.clip(image[:,:,i],0,threshold_pixel[i])
        return image
    else:
        return np.clip(image,0,1)

def read_exr(image_path, gamma = 1, percentile = 99):
    image = cv2.imread(image_path,  cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)  
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image = remove_bad_pixel(image,percentile = percentile)
    image = np.power(image,gamma) # gamma correction
    return image

def read_mask(mask_path):
    mask = cv2.imread(mask_path)
    if len(mask.shape) == 3 or mask.shape[2] == 3:
        mask = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
    return mask > 0

def normalize_pixel(image,mask):
    average_pixel = np.average(image[mask],axis=0) 
    #print(average_pixel) # for debug
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
    image = read_exr(packed_data['image_path'], packed_data['gamma'], packed_data['percentile'])
    image = normalize_pixel(image, packed_data['mask'])
    if packed_data['rotate_image'] == True:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE);
    plt.imsave(packed_data['output_path'],image)

def apply_masks(input_dir, output_dir, masks, num_thread = 8, rotate_image = True, gamma = 1.0, percentile = 99):
    packed_datas = []
    exr_files = os.listdir(input_dir)
    for exr_file in exr_files:
        if not exr_file.endswith('.exr'):
             continue
        packed_datas.append({
            'image_path': os.path.join(input_dir,exr_file),
            'mask': masks[exr_file.split('_')[0]],
            'output_path': os.path.join(output_dir, exr_file.split('.')[0] + '.png'),
            'rotate_image': rotate_image,
            'gamma': gamma,
            'percentile': percentile
        })
    with Pool(num_thread) as p:
        p.map(apply_mask,packed_datas)

def main(args):
    start_time = timer()
    if not os.path.exists(args.output):
        os.mkdir(args.output)
    print("using {} threads".format(args.thread))
    print("reading mask...")
    masks = get_masks(args.reference, args.thread)
    print("white balancing...")
    apply_masks(args.input, args.output, masks, args.thread, args.rotate_image, args.gamma, args.percentile)
    print("Running finished in {} seconds".format(timer() - start_time))
 
def entry_point():
    parser = argparse.ArgumentParser(
        description='deeparc2whitebalance.py - white balance the camera that output from deeparc')
    parser.add_argument(
        '-i',
        '--input',
        type=str,
        required=True,
        help='input exr file directory',
    )
    parser.add_argument(
        '-r',
        '--reference',
        type=str,
        required=True,
        help='mask to use as reference'
    ) 
    parser.add_argument(
        '-o',
        '--output',
        type=str,
        required= True,
        help='output directory')     
    parser.add_argument(
        '--thread',
        type=int,
        default=8,
        help='number of thread to use (default: 8)')
    parser.add_argument(
        '--percentile',
        type=float,
        default=99.0,
        help='percentile to remove bad pixel (default: 99.0)')
    parser.add_argument(
        '--gamma',
        type=float,
        default=1.0,
        help='gamma value for gamma correction (default: 1.0)')
    parser.add_argument(
        '--no-rotate',
        dest='rotate_image',
        action='store_false',
        help='do not rotate the output image'
    )    
    parser.set_defaults(rotate_image=True) 
    main(parser.parse_args())


if __name__ == "__main__":
    entry_point()