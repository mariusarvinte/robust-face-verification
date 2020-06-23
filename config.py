from model.network import Config

default_config = dict(
	content_dim=128,
	class_dim=256,

	content_std=1,
	content_decay=0.001,

	n_adain_layers=4,
	adain_dim=256,

	perceptual_loss=dict(
		layers=[2, 5, 8, 13, 18],
		weights=[1, 1, 1, 1, 1],
		scales=[64, ]
	),

	train=dict(
		batch_size=64,
		n_epochs=200
	),

	train_encoders=dict(
		batch_size=64,
		n_epochs=200
	)
)

# Create a dummy Lord object with the default Config
dummy_config = Config(
		img_shape=(64, 64, 3),
        n_imgs=None,
        n_classes=None,

		content_dim=default_config['content_dim'],
		class_dim=default_config['class_dim'],

		content_std=default_config['content_std'],
		content_decay=default_config['content_decay'],

		n_adain_layers=default_config['n_adain_layers'],
		adain_dim=default_config['adain_dim'],

		perceptual_loss_layers=default_config['perceptual_loss']['layers'],
		perceptual_loss_weights=default_config['perceptual_loss']['weights'],
		perceptual_loss_scales=default_config['perceptual_loss']['scales']
	)