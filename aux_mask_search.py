import numpy as np

# Function that creates a collection of allowed masks
def create_strided_masks(mask_size=20, stride=5, img_size=64):
    # Number of masks
    num_masks = (img_size-mask_size) // stride + 1
    # Leftover space
    leftover_space = 2
    # Empty masks
    out_masks = np.zeros((num_masks, num_masks, img_size, img_size, 3))
    
    # Populate in both dimensions
    for h_mask_idx in range(num_masks):
        for v_mask_idx in range(num_masks):
            out_masks[h_mask_idx, v_mask_idx,
                      (leftover_space+stride*h_mask_idx):(leftover_space+stride*h_mask_idx+mask_size),
                      (leftover_space+stride*v_mask_idx):(leftover_space+stride*v_mask_idx+mask_size), :] = 1.
            
    # Flatten
    out_masks = np.reshape(out_masks, (-1, img_size, img_size, 3))
    
    return out_masks

# Function that integrates gradient over a set of masks, picks top C candidates, 
# performs a forward pass, then select a final winner
def compute_gradient_magnitudes(grads, images, masks, model, anchors, minimize=True, C=5, img_size=64):
    # Get number of queries
    num_images = len(grads)
    # Output masks
    subwinner_masks  = np.zeros((num_images, C, img_size, img_size, 3))
    subwinner_images = np.zeros((num_images, C, img_size, img_size, 3))
    
    # Square grads
    squared_grads = np.square(grads)
    
    # For each image, integrate and sort
    for image_idx in range(num_images):
        # MSE trick
        grad_sums = np.sum(squared_grads[image_idx][None, :] * masks, axis=(-1, -2, -3))
        # Sort
        sorted_sums = np.flip(np.argsort(grad_sums))
        # Pick winners
        subwinner_masks[image_idx] = masks[sorted_sums[:C]]
        # Fill in images with neutral values in the mask locations
        subwinner_images[image_idx] = images[image_idx][None, :] * (1 - subwinner_masks[image_idx]) + 0.5 * subwinner_masks[image_idx]
        
    # If we only have one candidate, we are done
    if C == 1:
        winner_masks = np.squeeze(subwinner_masks)
    else:
        # Serialize, replicate anchor and compute loss, then reshape
        serial_anchors = np.repeat(anchors, C, axis=0)
        serial_images  = np.reshape(subwinner_images, (-1, img_size, img_size, 3))
        
        # Call model to get loss
        serial_losses = model.predict([serial_images, serial_anchors])
        
        # Reshape losses back
        matrix_losses = np.reshape(serial_losses, (num_images, C))
        
        # Pick final winners
        if minimize:
            winner_idx = np.argmin(matrix_losses, axis=-1)
        else:
            winner_idx = np.argmax(matrix_losses, axis=-1)
        
        # Downselect masks
        winner_masks = subwinner_masks[np.arange(num_images), winner_idx]
        
    return winner_masks