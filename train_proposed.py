import os
import numpy as np

import tensorflow as tf
from keras import backend as K

from keras.models import load_model
from model.network import AdaptiveInstanceNormalization

from keras.layers import Input, Dense, Flatten
from keras.layers import Lambda, Conv2D
from keras.models import Model
from keras.optimizers import Adam
from keras.constraints import UnitNorm
from keras.regularizers import l2
from keras.applications import VGG16

from keras.preprocessing.image import ImageDataGenerator
from random_eraser import apply_random_eraser_and_mask
from random_eraser import get_random_eraser_and_mask

import hdf5storage
from tqdm import tqdm
from sklearn.metrics import auc

def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 10
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)

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
tf.set_random_seed(1234)

# Directories of pretrained models/data
model_dir      = 'trained_models/lord/model/'
data_loc       = 'trained_models/lord/data/celeba_test.npz'
train_data_loc = 'trained_models/lord/data/celeba_vgg.npz'
cbk_loc        = 'trained_codebooks/one_sample_fixed.mat'
train_cbk_loc  = 'trained_codebooks/train_one_sample_fixed.mat'

# Load all data
all_data = np.load(train_data_loc)
x_d_all  = np.copy(all_data['imgs'] / 255.)
y_d_all  = np.copy(all_data['classes'])    
# Load test data
data      = np.load(data_loc)
x_d_test  = np.copy(data['imgs'] / 255.)
y_d_test  = np.copy(data['classes'])
# Rearrange y_test as ordinal classes (since absolute value of class doesn't matter)
_, y_d_test_ordinal = np.unique(y_d_test, return_inverse=True)

# Filter test data from training data
is_train = np.logical_not(np.isin(y_d_all, y_d_test))
x_d_train = np.copy(x_d_all[is_train])
y_d_train = np.copy(y_d_all[is_train])
# Free up memory
del all_data, x_d_all, y_d_all
# Rearrange y_train as ordinal classes (since absolute value of class doesn't matter)
_, y_d_train_ordinal = np.unique(y_d_train, return_inverse=True)

# Load model by parts
content_encoder = load_model(os.path.join(model_dir, 'content_encoder.h5py'))
class_encoder = load_model(os.path.join(model_dir, 'class_encoder.h5py'))
class_modulation = load_model(os.path.join(model_dir, 'class_modulation.h5py'))
generator = load_model(os.path.join(model_dir, 'generator.h5py'), custom_objects={
		'AdaptiveInstanceNormalization': AdaptiveInstanceNormalization})

# Predict content
# Train
train_content = content_encoder.predict(x_d_train)
# Test
test_content  = content_encoder.predict(x_d_test)

# Load modulation codebooks
contents                  = hdf5storage.loadmat(train_cbk_loc)
train_person_mod_codebook = contents['frozen_class_mod']
train_person_codebook     = contents['frozen_class']
contents                  = hdf5storage.loadmat(cbk_loc)
person_mod_codebook       = contents['frozen_class_mod']
person_codebook           = contents['frozen_class']

# Construct training and validation sets
np.random.seed(2020) # Current year
num_train_persons = 2000
num_val_persons   = 100 # Drawn from test persons
train_persons = np.random.choice(np.max(y_d_train_ordinal)+1, size=num_train_persons, replace=False)
val_persons   = np.random.choice(np.max(y_d_test_ordinal)+1, size=num_val_persons, replace=False)
x_train       = np.copy(x_d_train[np.isin(y_d_train_ordinal, train_persons)])
x_val         = np.copy(x_d_test[np.isin(y_d_test_ordinal, val_persons)])
y_train       = np.copy(y_d_train_ordinal[np.isin(y_d_train_ordinal, train_persons)])
y_val         = np.copy(y_d_test_ordinal[np.isin(y_d_test_ordinal, val_persons)])
c_train       = np.copy(train_content[np.isin(y_d_train_ordinal, train_persons)])
c_val         = np.copy(test_content[np.isin(y_d_test_ordinal, val_persons)])

# Once we pick validation persons, construct their clean reconstructions
x_match_val = generator.predict([c_val, person_mod_codebook[y_val]])
# Free up memory
del x_d_train, x_d_test, train_content, test_content


# Training parameters
batch_size    = 256
mining_steps  = 2
num_steps     = 20000
alpha         = 1e-3  # Weight decay coefficient
best_area_val = 0.
best_val_loss = 1e9
# Learning algorithm
trainer   = 'adam'
adv_steps = 10
adv_lr    = 16. / 255 # Pixels at once 
symmetrical_adv = True # Train symmetrically
# Architecture
latent_dim = 128
# Universal labels (for a single batch)
train_pair_labels = np.concatenate((np.ones(batch_size//2), np.zeros(batch_size//2)))[:, None]
val_pairs       = len(x_val)
val_pair_labels = np.concatenate((np.ones(val_pairs), np.zeros(val_pairs)))[:, None]

# Input tensors
input_img = Input(shape=(64, 64, 3))

# Dynamic architecture
# Load a VGG16
core_model = VGG16(input_shape=(64, 64, 3), include_top=False)
encoded = core_model(input_img)
# Feature layer
encoded = Flatten()(encoded)
encoded = Dense(latent_dim, activation='linear', kernel_constraint=UnitNorm())(encoded)

# Create shared model
shared_model = Model(input_img, encoded)

# Two input tensors
img_real = Input(shape=(64, 64, 3))
img_gen  = Input(shape=(64, 64, 3))

# Get features
features_real = shared_model(img_real)
features_gen  = shared_model(img_gen)
# Compute distance
sim_score = Lambda(euclidean_distance)([features_real, features_gen])

# Siamese model
model = Model([img_real, img_gen], sim_score)

# Optimizer
optimizer = Adam(lr=0.001, amsgrad=True)
# Compile
model.compile(optimizer, loss=contrastive_loss, metrics=['accuracy'])
# Apply L2 weight regularization post-factum
for layer in core_model.layers:
    if isinstance(layer, Conv2D) or isinstance(layer, Dense):
        layer.add_loss(lambda: l2(alpha)(layer.kernel))
    if hasattr(layer, 'bias_regularizer') and layer.use_bias:
        layer.add_loss(lambda: l2(alpha)(layer.bias))
            
# Instantiate cutout
eraser = get_random_eraser_and_mask(p=0.5, s_l=0.02, s_h=0.2, r_1=0.5, r_2=2.,
                                    v_l=0., v_h=1., pixel_level=True)        
# Instantiate augmentation generator
image_generator = ImageDataGenerator(width_shift_range=5,
                                     height_shift_range=5,
                                     horizontal_flip=True)

# Setup a graph for patch adversarial attacks
x_adv      = tf.placeholder(dtype=tf.float32, shape=(None, 64, 64, 3))
x_adv_pair = tf.placeholder(dtype=tf.float32, shape=(None, 64, 64, 3))
# Get features of both
adv_real_features = shared_model(x_adv)
adv_pair_features = shared_model(x_adv_pair)
# Loss function and its gradient
adv_loss = tf.norm(adv_real_features - adv_pair_features, axis=-1)
grad,    = tf.gradients(adv_loss, x_adv)

# Where to save weights
result_dir  = 'trained_models/proposed'
if not os.path.exists(result_dir):
    os.mkdir(result_dir)
weight_name = result_dir + '/steps%d_lr%.1f' % (adv_steps, adv_lr*255.)

# Granularity of AUC
num_points = 100
tpr_val    = np.zeros((num_steps, num_points))
fpr_val    = np.zeros((num_steps, num_points))
area_val   = np.zeros((num_steps,))
# Training/validation logs
train_loss_log = np.zeros((num_steps,))
val_loss_log   = np.zeros((num_steps,))

# Train for each batch
for step_idx in tqdm(range(num_steps)):
    # Draw a half batch of samples
    random_idx   = np.random.choice(len(x_train), size=batch_size//2, replace=False)
    # Augment and generate them with their correct class codebook
    x_match_real_half_batch = image_generator.flow(x_train[random_idx],
                                                   shuffle=False, batch_size=batch_size//2)[0]
    # Random erasure
    x_match_real_half_batch, x_real_mask_half_batch = apply_random_eraser_and_mask(eraser, x_match_real_half_batch)
    # Get content code and generate images with correct class codes
    real_content = content_encoder.predict(x_match_real_half_batch)
    x_match_gen_half_batch  = generator.predict([real_content, train_person_mod_codebook[y_train[random_idx]]])
    
    # Adversarial attack on positive pair
    if symmetrical_adv:
        if adv_steps > 0:
            # Find indices where patch augmentation is applied
            patch_attack_idx = np.where(np.sum(x_real_mask_half_batch, axis=(1, 2, 3)))[0]
        
            # Check if at least one such sample exists
            if len(patch_attack_idx) > 0:
                # Compute feature differences before adversarial attack - enable if manual verification is desired, but will slow down processing
#                diff_before = model.predict([x_match_real_half_batch[patch_attack_idx],
#                                             x_match_gen_half_batch[patch_attack_idx]])
                
                # Further minimize distance by adversarial attacks
                x_orig      = np.copy(x_match_real_half_batch[patch_attack_idx])
                x_mask_aug  = x_real_mask_half_batch[patch_attack_idx]
                for internal_step_idx in range(adv_steps):
                    # Get gradients and outputs
                    grad_np = session.run(grad, feed_dict={x_adv: x_orig,
                                                           x_adv_pair: x_match_gen_half_batch[patch_attack_idx]})
                    # Normalize, apply and clip
                    x_orig = np.clip(x_orig + adv_lr * np.sign(grad_np) * x_mask_aug, 0., 1.)
                
                # Compute feature differences after adversarial attack
#                diff_after = model.predict([x_orig, x_match_gen_half_batch[patch_attack_idx]])
                # Replace samples with adversarials
                x_match_real_half_batch[patch_attack_idx] = x_orig
    
    # Mine for hard candidates that use the same class vectors
    fake_person_idx = np.asarray([np.random.choice(np.where(np.logical_not(np.isin(y_train, y_train[random_idx[idx]])))[0],
                                                   size=mining_steps, replace=False) for idx in range(batch_size//2)]).flatten()
    
    # Generate fake images with the target's class codes
    fake_input_candidates = x_train[fake_person_idx]
    mod_input             = train_person_mod_codebook[np.mod(np.repeat(y_train[random_idx], mining_steps, axis=0),
                                                             len(train_person_mod_codebook)).astype(np.int)]
    
    # Augment all negative pairs and generate them
    fake_input_candidates_aug = image_generator.flow(fake_input_candidates,
                                                     shuffle=False, batch_size=batch_size//2*mining_steps)[0]
    # Random erasure - save the mask for potential attacks
    fake_input_candidates_aug, fake_erasure_mask = apply_random_eraser_and_mask(eraser, fake_input_candidates_aug)
    # Get content code and generate images with swapped class codes
    fake_content = content_encoder.predict(fake_input_candidates_aug)
    fake_output_candidates = generator.predict([fake_content, mod_input])
    
    # Get their similarity on input-output pairs
    fake_sim_candidates = model.predict([fake_input_candidates_aug, fake_output_candidates])
    # Reshape
    fake_sim_candidates        = np.reshape(fake_sim_candidates, (-1, mining_steps))
    fake_output_candidates     = np.reshape(fake_output_candidates, (batch_size//2, mining_steps, 64, 64, 3))
    fake_input_candidates_aug  = np.reshape(fake_input_candidates_aug, (batch_size//2, mining_steps, 64, 64, 3))
    fake_masks                 = np.reshape(fake_erasure_mask, (batch_size//2, mining_steps, 64, 64, 3))
    # Pick closest pairs
    fake_idx = np.argmin(fake_sim_candidates, axis=-1)
    
    # Assign the other half of batches
    x_fake_real_half_batch = fake_input_candidates_aug[np.arange(batch_size//2), fake_idx]
    x_fake_mask_half_batch = fake_masks[np.arange(batch_size//2), fake_idx]
    x_fake_gen_half_batch  = fake_output_candidates[np.arange(batch_size//2), fake_idx]
    
    if adv_steps > 0:
        # Find indices where patch augmentation is applied
        patch_attack_idx = np.where(np.sum(x_fake_mask_half_batch, axis=(1, 2, 3)))[0]
    
        # Check if at least one such sample exists
        if len(patch_attack_idx) > 0:
            # Compute feature differences before adversarial attack
#            diff_before = model.predict([x_fake_real_half_batch[patch_attack_idx],
#                                         x_fake_gen_half_batch[patch_attack_idx]])
#            
            # Further minimize distance by adversarial attacks
            x_orig = np.copy(x_fake_real_half_batch[patch_attack_idx])
            z_class_aug = train_person_mod_codebook[y_train[random_idx]][patch_attack_idx]
            x_mask_aug  = x_fake_mask_half_batch[patch_attack_idx]
            for internal_step_idx in range(adv_steps):
                # Get gradients and outputs
                grad_np = session.run(grad, feed_dict={x_adv: x_orig,
                                            x_adv_pair: x_fake_gen_half_batch[patch_attack_idx]})
                # Normalize, apply and clip
                x_orig = np.clip(x_orig - adv_lr * np.sign(grad_np) * x_mask_aug, 0., 1.)
            
            # Compute feature differences after adversarial attack
#            diff_after = model.predict([x_orig, x_fake_gen_half_batch[patch_attack_idx]])
            # Replace samples with adversarials
            x_fake_real_half_batch[patch_attack_idx] = x_orig
        
        # Construct batches
        x_real_batch = np.concatenate((x_match_real_half_batch, x_fake_real_half_batch), axis=0)
        x_gen_batch = np.concatenate((x_match_gen_half_batch, x_fake_gen_half_batch), axis=0)
    else:
        # Construct batches
        x_real_batch = np.concatenate((x_match_real_half_batch, x_fake_real_half_batch), axis=0)
        x_gen_batch = np.concatenate((x_match_gen_half_batch, x_fake_gen_half_batch), axis=0)

    # Train on batch
    train_loss, train_acc = model.train_on_batch([x_real_batch, x_gen_batch], train_pair_labels)
    
    # Validate periodically
    if np.mod(step_idx, 50) == 0:
        # For each person, sample another person from the same class
        sampled_real_idx = np.asarray([np.random.choice(np.setdiff1d(np.where(y_val == y_val[idx])[0], idx)) for idx in range(len(y_val))])
        # For each person, sample another person
        sampled_fake_idx = np.asarray([np.random.choice(np.where(y_val != y_val[idx])[0]) for idx in range(len(y_val))])
        # Create merged vectors
        x_real_val  = np.concatenate((x_val, x_val), axis=0)
        x_fake_val  = np.concatenate((x_val[sampled_real_idx], x_val[sampled_fake_idx]), axis=0)
        
        # Predict
        val_loss, val_acc = model.evaluate([x_real_val, x_fake_val], val_pair_labels)
        print('Step %d. Val. loss = %.3f, Val. acc. = %.3f' % (step_idx, val_loss, val_acc))
        # Verbose
        print('Step %d. Train loss = %.3f, Train acc. = %.3f' % (step_idx, train_loss, train_acc))
        # Directly get similarities
        val_sim = model.predict([x_real_val, x_fake_val])
        real_sim, fake_sim = np.split(val_sim, [len(val_sim)//2])
        # Compute AUC ad-hoc
        min_sim = np.minimum(np.min(real_sim), np.min(fake_sim))
        max_sim = np.maximum(np.max(real_sim), np.max(fake_sim))
        thresholds = np.linspace(min_sim, max_sim, num_points)
        for idx, threshold in enumerate(thresholds):
            tpr_val[step_idx, idx] = np.mean(real_sim < threshold)
            fpr_val[step_idx, idx] = np.mean(fake_sim < threshold)
        # Compute AUC
        area_val[step_idx] = auc(fpr_val[step_idx], tpr_val[step_idx])
        print('Step %d. AUC = %.3f' % (step_idx, area_val[step_idx]))
        
        # Save best model according to AUC
        if area_val[step_idx] > best_area_val:
            shared_model.save_weights(weight_name + '_auc.h5')
            best_area_val = area_val[step_idx]
        # Save best model according to validation loss
        if val_loss < best_val_loss:
            shared_model.save_weights(weight_name + '_loss.h5')
            best_val_loss = val_loss
        # Save latest weights always
        shared_model.save_weights(weight_name + '_last.h5')
            
        # Store in logs
        train_loss_log[step_idx] = train_loss
        val_loss_log[step_idx]   = val_loss
        # Save periodically
        hdf5storage.savemat(weight_name + '_logs.mat', {'train_loss_log': train_loss_log,
                                                        'val_loss_log': val_loss_log},
            truncate_existing=True)
