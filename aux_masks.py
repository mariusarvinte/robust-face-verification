import numpy as np
import cv2
import matplotlib.image as mpimg

# Function that returns various numpy arrays with masks
def gen_mask(batch_size, mask_style, mask_size):
    # What kind of mask
    if mask_style == 'corner_patch':
        # Always assumed to be in upper left
        mask_np = np.zeros((batch_size, 64, 64, 3))
        mask_np[:, :mask_size, -mask_size:, :] = 1.
        
    elif mask_style == 'eye_patch' or mask_style == 'universal_patch':
        # Fixed size, specific for average eye location in aligned CelebA
        eyes_coordsx = [26, 38]
        eyes_coordsy = [16, 48]
        mask_np  = np.zeros((batch_size, 64, 64, 3))
        mask_np[:, eyes_coordsx[0]:eyes_coordsx[1],
                    eyes_coordsy[0]:eyes_coordsy[1], :] = 1
                
    elif mask_style == 'eyeglasses':
        # Fixed size, specific for average eye location in aligned CelebA
        eyes_coordsx = [24, 40]
        eyes_coordsy = [16, 48]
        # Load from file
        image_full = mpimg.imread('eyeglasses-310516_640.png')
        image = image_full[:, :, -1] # Extract the gray component
        # Reshape
        image = cv2.resize(image, dsize=(32, 16))
        # Convert to binary mask
        image = (image > 0.1).astype(np.float32)
        # Pad up to 64x64
        image = np.pad(image, ((eyes_coordsx[0], 64-eyes_coordsx[1]),
                               (eyes_coordsy[0], 64-eyes_coordsy[1])))
        # Replicate
        mask_np = np.repeat(np.repeat(image[:, :, None], 3, axis=-1)[None, :], batch_size, axis=0)
        
    elif mask_style == 'all':
        # A large portion of the image is enabled
        mask_np = np.ones((batch_size, 64, 64, 3))
        
    return mask_np

# Function that returns a grid of masks that superimpose the faces in CelebA
def gen_candidate_masks():
    # Hardcoded
    mask_start  = 15
    mask_size   = 10
    mask_stride = 5
    num_masks   = 5
    # Outputs
    output_masks = np.zeros((num_masks, num_masks, 64, 64, 3))
    
    # For each location
    for h_mask_idx in range(num_masks):
        for v_mask_idx in range(num_masks):
            output_masks[h_mask_idx, v_mask_idx,
                         mask_start+h_mask_idx*mask_stride:mask_start+h_mask_idx*mask_stride+mask_size,
                         mask_start+v_mask_idx*mask_stride:mask_start+v_mask_idx*mask_stride+mask_size,
                         :] = 1.
                         
    # Serialize
    output_masks = np.reshape(output_masks, (-1, 64, 64, 3))
    return output_masks