import numpy as np

import os
import tensorflow as tf

from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense, Flatten
from keras.constraints import UnitNorm
from keras.applications import VGG16

from sklearn.metrics import auc
from sklearn.metrics import average_precision_score

import hdf5storage
from tqdm import tqdm

# GPU allocation
K.clear_session()
tf.reset_default_graph()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"] = "0";
# Tensorflow memory allocation
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 1.
session = tf.Session(config=config)
K.tensorflow_backend.set_session(session)
K.set_learning_phase(False)

# Directories of pretrained models/data
model_dir = 'trained_models/lord/model/'
data_loc  = 'trained_models/lord/data/celeba_test.npz'
cbk_loc   = 'trained_codebooks/one_sample_fixed.mat'

# Load data
data      = np.load(data_loc)
x_d_test  = np.copy(data['imgs'] / 255.)
y_d_test  = np.copy(data['classes'])
# Rearrange y_test as ordinal classes (since absolute value of class doesn't matter)
_, y_d_test_ordinal = np.unique(y_d_test, return_inverse=True)

# Instantiate and load VGGFace with VGG16 core
latent_dim = 128
input_img  = Input(shape=(64, 64, 3))
core_model = VGG16(input_shape=(64, 64, 3), include_top=False)
encoded    = core_model(input_img)
# Feature layer
encoded = Flatten()(encoded)
encoded = Dense(latent_dim, activation='linear', kernel_constraint=UnitNorm())(encoded)
# Create shared model
model = Model(input_img, encoded)

# Load weights
core_folder    = 'trained_models/proposed'
core_weights   = 'steps16_lr10.0_last' 
target_weights = '%s/%s.h5' % (core_folder, core_weights)
model.load_weights(target_weights)

# Which results do we load
mask_style = 'eyeglasses'
mask_size  = 10
if mask_style == 'corner_patch' or mask_style == 'frame':
    mask_name = '%s%d' % (mask_style, mask_size)
else:
    mask_name = mask_style
# Repetition parameters
random_start    = True
# Does the defender use mirroring?
def_mirroring   = True
# Does the defender pick the anchor?
def_anchor_pick = True

# Regenerate targets (or just look at subfolders)
np.random.seed(2020)
num_persons = 30
target_person_list = np.random.choice(np.max(y_d_test_ordinal)+1, replace=False, size=num_persons)
# How many images to evaluate per person
target_limit = 100 # Large means use everything
# AUC curve granularity
num_thresholds = 1000

# Get result directory
meta_folder    = 'paper_attacks_%s' % (mask_name)
submeta_folder = 'mirror%d_anchor%d_random%d' % (def_mirroring, def_anchor_pick, random_start)
result_dir     = '%s/%s/%s/%s' % (meta_folder, submeta_folder, core_folder, core_weights)

# Store feature losses
global_target_loss   = []
global_int_loss      = [] 
global_adv_loss      = []
global_indirect_loss = []

# Reload all results
for person_idx, target_person in enumerate(target_person_list):
    # Split person/others
    x_target_person = x_d_test[y_d_test_ordinal == target_person]
    x_other_person  = x_d_test[y_d_test_ordinal != target_person]
    # Number of images of the target person
    num_samples_target = len(x_target_person)
    
    # If there are too few images of the target, skip them
    if num_samples_target < 10:
        print('Skipping target %d, too few samples' % target_person)
        continue
    
    # Statistics for each person
    local_target_mean = []
    
    if def_anchor_pick:
        # Pick each sample as the template
        for sample_idx in tqdm(range(np.minimum(num_samples_target, target_limit))):
            # Compare all images with the template
            x_target_anchor = x_target_person[sample_idx]
            x_target_pair   = x_target_person[np.logical_not(np.isin(np.arange(num_samples_target), sample_idx))]
            # Get their features
            x_target_anchor_features = model.predict(x_target_anchor[None, :])
            x_target_pair_features   = model.predict(x_target_pair)
            # Pairwise distances
            target_feature_loss = np.sum(np.square(x_target_anchor_features - x_target_pair_features), axis=-1)
            # Compute and store average distance
            target_feature_mean = np.mean(target_feature_loss)
            # Store separately
            local_target_mean.append(target_feature_mean)
            
        # Once we went through all samples, pick the best anchor
        best_anchor_idx = np.argmin(local_target_mean)
        
    else:
        # Use the first (random) sample as anchor
        best_anchor_idx = 0
        
    # Re-split anchor and others
    x_target_anchor = x_target_person[best_anchor_idx]
    x_target_pair   = x_target_person[np.logical_not(np.isin(np.arange(num_samples_target), best_anchor_idx))]
    # Pick a number of random intruders
    x_random_other = x_other_person[np.random.choice(len(x_other_person), size=128, replace=False)]
    
    # Get features of anchor
    x_target_anchor_features = model.predict(x_target_anchor[None, :])
    # Get features of regular- and mirrored other samples
    x_target_pair_features = model.predict(x_target_pair)
    if def_mirroring:
        x_target_mirror_features = model.predict(np.flip(x_target_pair, axis=-2))
        # Average pairwise distance
        target_mirror_loss = (np.sum(np.square(x_target_anchor_features - x_target_pair_features), axis=-1) + \
        np.sum(np.square(x_target_anchor_features - x_target_mirror_features), axis=-1)) / 2
    else:
        target_mirror_loss = np.sum(np.square(x_target_anchor_features - x_target_pair_features), axis=-1)
    
    # Same for intruders
    x_other_pair_features = model.predict(x_random_other)
    if def_mirroring:
        x_other_mirror_features = model.predict(np.flip(x_random_other, axis=-2))
        # Average pairwise distance
        other_mirror_loss = (np.sum(np.square(x_target_anchor_features - x_other_pair_features), axis=-1) + \
        np.sum(np.square(x_target_anchor_features - x_other_mirror_features), axis=-1)) / 2
    else:
        other_mirror_loss = np.sum(np.square(x_target_anchor_features - x_other_pair_features), axis=-1)  
           
    # Store losses themselves
    global_target_loss.append(target_mirror_loss)
    global_int_loss.append(other_mirror_loss)
    
    # Get local directory
    local_dir = result_dir + '/target%d_sample%d' % (target_person, best_anchor_idx)
    try:
        contents = hdf5storage.loadmat(local_dir + '/attack.mat')
        if mask_style == 'eyeglasses':
            # Load the results of the indirect attack (only for eyeglasses currently, can be anything else)
            try:
                contents_indirect = hdf5storage.loadmat('paper_attacks_indirect_eyeglasses/%s/%s/%s/target%d_sample%d/attack.mat' % (
                        submeta_folder, core_folder, core_weights, target_person, best_anchor_idx))
                adv_indirect_loss = contents_indirect['adv_true_feature_loss']
                global_indirect_loss.append(adv_indirect_loss)

            except:
                print('Skipping target %d! No indirect attack found' % target_person)
    except:
        print('Skipping target %d!' % target_person)
        continue
    # Unwrap
    adv_feature_loss = contents['adv_feature_loss']
    
    # Store losses themselves
    global_adv_loss.append(adv_feature_loss)
    
# Compute AUC based on a single threshold
global_target_loss   = np.hstack(global_target_loss)
global_int_loss      = np.hstack(global_int_loss)
global_adv_loss      = np.hstack(global_adv_loss)
try:
    global_indirect_loss = np.hstack(global_indirect_loss)
    # Compute the minimum loss between the two attacks
    global_worst_loss    = np.minimum(global_adv_loss, global_indirect_loss)
except:
    print('Indirect mode deactivated')

# Universal intruder AUC
min_dist = np.minimum(np.min(global_target_loss), np.min(global_int_loss))
max_dist = np.maximum(np.max(global_target_loss), np.max(global_int_loss))
thresholds = np.linspace(min_dist*0.999, max_dist*1.001, num_thresholds)
# Manual loop
uni_int_fpr, uni_int_tpr = np.zeros((num_thresholds,)), np.zeros((num_thresholds,))
uni_int_fnr = np.zeros((num_thresholds,))
for idx, threshold in enumerate(thresholds):
    uni_int_fpr[idx] = np.mean(global_target_loss > threshold)
    uni_int_tpr[idx] = np.mean(global_int_loss > threshold)
    uni_int_fnr[idx] = np.mean(global_int_loss <= threshold)
# AUC-ROC
universal_int_area = auc(uni_int_fpr, uni_int_tpr)
# Compute precision-recall
int_prec   = uni_int_tpr / (uni_int_tpr + uni_int_fpr)
int_recall = uni_int_tpr / (uni_int_tpr + uni_int_fnr)
int_prec, int_recall = int_prec[:-1], int_recall[:-1] # Remove 0/0 point
universal_int_aupr = auc(int_recall, int_prec)
    
# Universal adversarial AUC
min_dist = np.minimum(np.min(global_target_loss), np.min(global_adv_loss))
max_dist = np.maximum(np.max(global_target_loss), np.max(global_adv_loss))
thresholds = np.linspace(min_dist*0.999, max_dist*1.001, num_thresholds)
# Manual loop
uni_adv_fpr, uni_adv_tpr = np.zeros((num_thresholds,)), np.zeros((num_thresholds,))
uni_adv_fnr = np.zeros((num_thresholds,))
for idx, threshold in enumerate(thresholds):
    uni_adv_fpr[idx] = np.mean(global_target_loss > threshold)
    uni_adv_tpr[idx] = np.mean(global_adv_loss > threshold)
    uni_adv_fnr[idx] = np.mean(global_adv_loss <= threshold)
# AUC-ROC
universal_adv_area = auc(uni_adv_fpr, uni_adv_tpr)
# Compute precision-recall
adv_prec   = uni_adv_tpr / (uni_adv_tpr + uni_adv_fpr)
adv_recall = uni_adv_tpr / (uni_adv_tpr + uni_adv_fnr)
adv_prec, adv_recall = adv_prec[:-1], adv_recall[:-1] # Remove 0/0 point
universal_adv_aupr  = auc(adv_recall[:-1], adv_prec[:-1])

# Convert distances to detection logits (with some scaling)
target_logits = 1 - np.exp(-global_target_loss / 20.)
int_logits    = 1 - np.exp(-global_int_loss / 20.)
adv_logits    = 1 - np.exp(-global_adv_loss / 20.)

# Compute average precision score
ap_int = average_precision_score(np.concatenate((np.zeros((len(target_logits))),
                                                 np.ones((len(int_logits)))), axis=0),
            np.concatenate((target_logits, int_logits), axis=0))
ap_adv = average_precision_score(np.concatenate((np.zeros((len(target_logits))),
                                     np.ones((len(adv_logits)))), axis=0),
            np.concatenate((target_logits, adv_logits), axis=0))
                    
