import numpy as np

import os
import tensorflow as tf

from matplotlib import pyplot as plt
plt.ioff()

from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense, Flatten
from keras.constraints import UnitNorm
from keras.applications import VGG16

from aux_masks import gen_mask

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
data_loc  = 'trained_models/lord/data/celeba_test.npz'

# Load data
data      = np.load(data_loc)
x_d_test  = np.copy(data['imgs'] / 255.)
y_d_test  = np.copy(data['classes'])
# Rearrange y_test as ordinal classes (since absolute value of class doesn't matter)
_, y_d_test_ordinal = np.unique(y_d_test, return_inverse=True)

# Instantiate and load VGGFace wit VGG16 core
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

# Attack parameters
batch_size     = 100 # Intruders we train on
learning_rate  = 5e-3
num_iterations = 5000
# Repetition parameters
random_start   = False
num_restarts   = 5
# Instant overwrite
if not random_start:
    num_restarts = 1
# Mask type/size/style
mask_size  = 10
mask_style = 'universal_patch'
if mask_style == 'universal_patch' or mask_style == 'frame':
    mask_name = '%s%d' % (mask_style, mask_size)
else:
    mask_name = mask_style
# Does the defender use mirroring?
def_mirroring = True
# Does the defender pick the anchor?
def_anchor_pick = True
# Does the defender use a universal threshold?
def_universal_threshold = False # Irrelevant for the attack itself

# Generate the mask
mask_np = gen_mask(batch_size, mask_style, mask_size)[0][np.newaxis]

# Tensors
if random_start:
    delta = tf.Variable(tf.random.uniform(shape=(1, 64, 64, 3),
                                          minval=-0.5, maxval=0.5, dtype=tf.float32),
    dtype=np.float32)
else:
    delta = tf.Variable(np.zeros((1, 64, 64, 3)), dtype=np.float32)
    
# Input tensors
x_input           = tf.placeholder(dtype=np.float32, shape=(batch_size, 64, 64, 3))
x_target_features = tf.placeholder(dtype=np.float32, shape=(batch_size, latent_dim))
# Tensor mask
loss_mask = tf.constant(mask_np, dtype=np.float32)

# Add adversarial noise
x_adv = tf.tanh(x_input + tf.multiply(loss_mask, delta))/2 + 1/2
# Mirror image
x_adv_mirror = tf.image.flip_left_right(x_adv)

# Get features
adv_features    = model(x_adv)
mirror_features = model(x_adv_mirror)

# Feature loss
if def_mirroring:
    feature_loss = (tf.reduce_sum(tf.square(adv_features - x_target_features), axis=-1) +\
    tf.reduce_sum(tf.square(mirror_features - x_target_features), axis=-1)) / 2
else:
    feature_loss = tf.reduce_sum(tf.square(adv_features - x_target_features), axis=-1)
# Deviation loss
dev_loss = tf.reduce_sum(tf.square(tf.tanh(x_input)/2+1/2 - x_adv), axis=(1, 2, 3))

# Merge into single loss
target_loss = tf.reduce_sum(feature_loss) 

# Adam optimizer
start_vars   = set(x.name for x in tf.global_variables())
optimizer    = tf.train.AdamOptimizer(learning_rate=learning_rate)
trainer      = optimizer.minimize(target_loss, var_list=[delta])
end_vars     = tf.global_variables()
new_vars     = [x for x in end_vars if x.name not in start_vars]
init         = tf.variables_initializer(var_list=[delta]+new_vars)

# Create result directory
meta_folder    = 'paper_attacks_%s' % (mask_name)
submeta_folder = 'mirror%d_anchor%d_random%d' % (def_mirroring, def_anchor_pick, random_start)
result_dir     = '%s/%s/%s/%s' % (meta_folder, submeta_folder, core_folder, core_weights)
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

# Pick a target
np.random.seed(2020)
num_persons = 10
target_person_list = np.random.choice(np.max(y_d_test_ordinal)+1, replace=False, size=num_persons)
for target_person in target_person_list:
    # Wipe variables
    session.run(init)
    local_target_mean = []
    best_test_loss    = 1e9
    
    # Seed results
    np.random.seed(target_person)
    # Split person/others
    x_target_person = x_d_test[y_d_test_ordinal == target_person]
    x_others_person = x_d_test[y_d_test_ordinal != target_person]
    # Number of images of the target person
    num_samples_target = len(x_target_person)
    
    # If there are too few images of the target, skip them
    if num_samples_target < 10:
        print('Skipping target %d, too few samples' % target_person)
        continue
    
    # Does the defender pick their anchor?
    if def_anchor_pick:
        # Pick each sample as the template - this exactly emulates the defender
        for sample_idx in tqdm(range(num_samples_target)):
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
        anchor_idx = np.argmin(local_target_mean)
    else:
        # Pick the first (equivalent to a random) sample to use as anchor
        anchor_idx = 0
        
    # Create a local directory
    local_dir = result_dir + '/target%d_sample%d' % (target_person, anchor_idx)
    if not os.path.exists(local_dir):
        os.mkdir(local_dir)
        
    # Use exactly the template the defender uses
    x_target_attack = x_target_person[anchor_idx]
    # Get their features
    x_target_real_features = model.predict(x_target_person)
    
    # Fetch two batches image, one for training and one for testing
    others_merged_idx = np.random.choice(x_others_person.shape[0], size=2*batch_size, replace=False)
    others_idx      = others_merged_idx[:batch_size]
    others_test_idx = others_merged_idx[batch_size:]
    # Construct feed dictionary
    feed_dict = {x_input: np.arctanh((x_others_person[others_idx] - 1/2) * 2 * 0.999999),
                 x_target_features: np.repeat(x_target_real_features[anchor_idx][None, :], batch_size, axis=0)}
    
    # Store losses across (potential) multiple runs
    test_feature_loss   = np.zeros((num_restarts, num_iterations, batch_size))
    feature_loss_matrix = np.zeros((num_restarts, batch_size))
    dev_loss_matrix     = np.zeros((num_restarts, batch_size))
    x_adv_matrix        = np.zeros((num_restarts, batch_size, 64, 64, 3))
    # For each repetition
    for rep_idx in range(num_restarts):
        # Wipe graph
        session.run(init)
        # Verbose
        print('Running restart %d.' % rep_idx)
        # Run attack
        for step_idx in range(num_iterations):
            _, delta_np, feature_loss_np, dev_loss_np, x_adv_np = session.run([trainer, delta, feature_loss,
                                                                               dev_loss, x_adv], 
        feed_dict=feed_dict)
            
            # Apply the patch on top of the test persons
            others_patched = np.tanh(np.arctanh((x_others_person[others_test_idx] - 1/2) * 2 * 0.99999)*(1 - mask_np) + delta_np * mask_np) / 2 + 1/2
            # Get their loss
            test_feature_loss[rep_idx, step_idx] = session.run(feature_loss, {x_adv: others_patched,
                    x_target_features: np.repeat(x_target_real_features[anchor_idx][None, :], batch_size, axis=0)})
        
            # Compute the average and replace the perturbation if it is the best
            instant_mean_test_loss = np.mean(test_feature_loss[rep_idx, step_idx])
            if instant_mean_test_loss < best_test_loss:
                best_delta        = delta_np
                best_feature_loss = test_feature_loss[rep_idx, step_idx]
                best_test_loss    = instant_mean_test_loss
            
            # Verbose
            if np.mod(step_idx+1, num_iterations//10) == 0:
                print('Iteration %d, Feature MSE %.3f, Test Feature MSE %.3f, Deviation MSE %.3f' % (step_idx,
                      np.mean(feature_loss_np), instant_mean_test_loss, np.mean(dev_loss_np)))
        
        # Store in meta arrays
        feature_loss_matrix[rep_idx] = feature_loss_np
        x_adv_matrix[rep_idx] = x_adv_np
    
    # Compute average loss
    avg_test_feature_loss = np.mean(test_feature_loss, axis=-1)
    
    # After all repetitions, pick best solutions
    winner_idx = np.argmin(feature_loss_matrix, axis=0)
    # Instantly overwrite
    feature_loss_np = feature_loss_matrix[winner_idx, np.arange(batch_size)]
    dev_loss_np     = dev_loss_matrix[winner_idx, np.arange(batch_size)]
    x_adv_np        = x_adv_matrix[winner_idx, np.arange(batch_size)]
            
    # Plot
    plt.figure()
    plt.suptitle('Feature MSE = %.3f, Deviation MSE = %.3f' % (feature_loss_np[0], dev_loss_np[0]))
    plt.subplot(2, 2, 1); plt.imshow(x_others_person[others_idx[0]]); plt.axis('off'); plt.title('Intruder Original')
    plt.subplot(2, 2, 2); plt.imshow(x_adv_np[0]); plt.axis('off'); plt.title('Intruder Adversarial')
    plt.tight_layout(rect=[0, 0., 1, 0.9])
    plt.savefig(local_dir + '/attack.png', dpi=300)
    # Save data
    hdf5storage.savemat(local_dir + '/attack.mat', {'x_adv_np': x_adv_np,
                                                    'x_others_person': x_others_person[others_idx],
                                                    'adv_feature_loss': feature_loss_np,
                                                    'dev_loss': dev_loss_np,
                                                    'avg_test_feature_loss': avg_test_feature_loss,
                                                    'best_delta': best_delta,
                                                    'best_feature_loss': best_feature_loss},
            truncate_existing=True)
    plt.close()