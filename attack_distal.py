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

# Attack parameters
batch_size     = 8 # Number of restarts
num_thresholds = 1000 # For AUC
learning_rate  = 1e-2
num_iterations = 2000
mask_size      = 10
mask_style     = 'all'
if mask_style == 'corner_patch':
    mask_name = '%s%d' % (mask_style, mask_size)
else:
    mask_name = mask_style

# Does the defender use mirroring?
def_mirroring = True
# Does the defender pick the anchor?
def_anchor_pick = True

# Generate the mask
mask_np = gen_mask(batch_size, mask_style, mask_size)

# Tensors
delta        = tf.Variable(tf.zeros(shape=(batch_size, 64, 64, 3), dtype=tf.float32),
    dtype=np.float32)
x_input      = tf.placeholder(dtype=np.float32, shape=(batch_size, 64, 64, 3))
# This is a tensor for the real target, in case the attacker wants to copy him directly
x_target     = tf.placeholder(dtype=np.float32, shape=(batch_size, 64, 64, 3))
# Tensor mask
loss_mask    = tf.constant(mask_np, dtype=np.float32)

# Add adversarial noise
x_adv = tf.tanh(x_input + tf.multiply(loss_mask, delta))/2 + 1/2
# Mirror image
x_adv_mirror = tf.image.flip_left_right(x_adv)

# Get features
adv_features    = model(x_adv)
mirror_features = model(x_adv_mirror)
target_features = model(x_target)

# Feature loss
if def_mirroring:
    feature_loss = (tf.reduce_sum(tf.square(adv_features - target_features), axis=-1) +\
    tf.reduce_sum(tf.square(mirror_features - target_features), axis=-1)) / 2
else:
    feature_loss = tf.reduce_sum(tf.square(adv_features - target_features), axis=-1)
    
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
submeta_folder = 'mirror%d_anchor%d' % (def_mirroring, def_anchor_pick)
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
    
    # Seed results
    np.random.seed(target_person)
    # Split person/others
    x_target_person = x_d_test[y_d_test_ordinal == target_person]
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
            x_target_pair_features  = model.predict(x_target_pair)
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
    
    # Inject noise
    x_others_person = np.random.randint(low=0, high=256, size=(batch_size, 64, 64, 3)) / 255.
    # Construct feed dictionary
    feed_dict = {x_input: np.arctanh((x_others_person - 1/2) * 2 * 0.999999),
                 x_target: np.repeat(x_target_attack[None, :], batch_size, axis=0)}
    
    # Run attack
    for step_idx in range(num_iterations):
        _, feature_loss_np, dev_loss_np, x_adv_np = session.run([trainer, feature_loss,
                                                                 dev_loss, x_adv], 
    feed_dict=feed_dict)
        
        # Verbose
        if np.mod(step_idx+1, num_iterations//10) == 0:
            print('Iteration %d, Feature MSE %.3f, Deviation MSE %.3f' % (step_idx,
                  np.mean(feature_loss_np), np.mean(dev_loss_np)))
            
    # Plot
    plt.figure()
    plt.suptitle('Feature MSE = %.3f, Deviation MSE = %.3f' % (feature_loss_np[0], dev_loss_np[0]))
    plt.subplot(2, 2, 1); plt.imshow(x_others_person[0]); plt.axis('off'); plt.title('Intruder Original')
    plt.subplot(2, 2, 2); plt.imshow(x_adv_np[0]); plt.axis('off'); plt.title('Intruder Adversarial')
    plt.tight_layout(rect=[0, 0., 1, 0.9])
    plt.savefig(local_dir + '/attack.png', dpi=300)
    # Save data
    hdf5storage.savemat(local_dir + '/attack.mat', {'x_adv_np': x_adv_np,
                                                    'x_target_attack': x_target_attack,
                                                    'x_others_person': x_others_person,
                                                    'adv_feature_loss': feature_loss_np,
                                                    'dev_loss': dev_loss_np},
            truncate_existing=True)
    plt.close()

# Reproduce Figure 4 from paper
plt.figure()
plt.tight_layout()
num_images = 8
person_idx = 0 

# Parse code is here
for target_person in target_person_list:
    local_target_mean = []
    # Split person/others
    x_target_person = x_d_test[y_d_test_ordinal == target_person]
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
            x_target_pair_features  = model.predict(x_target_pair)
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
    
    # Plot the target image and a random output
    # Get local directory
    local_dir = result_dir + '/target%d_sample%d' % (target_person, anchor_idx)
    # Load data
    contents = hdf5storage.loadmat(local_dir + '/vggface_passthrough_attack.mat')
        
    plt.subplot(2, num_images, person_idx+1)
    plt.imshow(contents['x_target_attack']); plt.axis('off')
    plt.subplot(2, num_images, person_idx+1+num_images)
    plt.imshow(contents['x_adv_np'][-1]); plt.axis('off')
    
    # Increment
    person_idx = person_idx + 1

plt.subplots_adjust(wspace=0.1, hspace=0.)
plt.savefig('distals.png', dpi=300)
plt.close()