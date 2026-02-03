import tensorflow as tf
from tensorflow.keras import layers, Model


class ResBlock(layers.Layer):
    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = layers.Conv2D(filters, 3, padding="same")
        self.gn1 = layers.GroupNormalization(groups=32)
        self.conv2 = layers.Conv2D(filters, 3, padding="same")
        self.gn2 = layers.GroupNormalization(groups=32)
        self.act = layers.Activation("swish")

    def call(self, x):
        residual = x
        x = self.gn1(x)
        x = self.act(x)
        x = self.conv1(x)
        x = self.gn2(x)
        x = self.act(x)
        x = self.conv2(x)
        return x + residual


class AttentionBlock(layers.Layer):
    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.gn = layers.GroupNormalization(groups=32)
        self.q = layers.Dense(filters)
        self.k = layers.Dense(filters)
        self.v = layers.Dense(filters)
        self.proj = layers.Dense(filters)

    def call(self, x):
        h, w, c = x.shape[1], x.shape[2], x.shape[3]
        shortcut = x
        x = self.gn(x)

        # Flatten and transpose for attention
        q = self.q(layers.Reshape((-1, c))(x))
        k = self.k(layers.Reshape((-1, c))(x))
        v = self.v(layers.Reshape((-1, c))(x))

        # Scale dot-product attention
        attn = tf.matmul(q, k, transpose_b=True)
        attn = attn * tf.math.rsqrt(tf.cast(c, tf.float32))
        attn = tf.nn.softmax(attn, axis=-1)

        out = tf.matmul(attn, v)
        out = self.proj(out)
        out = layers.Reshape((h, w, c))(out)

        return shortcut + out


def build_vae(img_height, img_width, img_channels, latent_dim=4):
    # Encoder
    encoder_inputs = layers.Input(shape=(img_height, img_width, img_channels))

    x = layers.Conv2D(128, 3, padding="same")(encoder_inputs)

    # Downsample 1: 256 -> 128
    x = layers.Conv2D(128, 3, strides=2, padding="same")(x)
    x = ResBlock(128)(x)
    x = ResBlock(128)(x)

    # Downsample 2: 128 -> 64
    x = layers.Conv2D(256, 3, strides=2, padding="same")(x)
    x = ResBlock(256)(x)
    x = ResBlock(256)(x)

    # Downsample 3: 64 -> 32
    x = layers.Conv2D(512, 3, strides=2, padding="same")(x)
    x = ResBlock(512)(x)
    x = AttentionBlock(512)(x)
    x = ResBlock(512)(x)

    # Latent space projection
    x = layers.GroupNormalization(groups=32)(x)
    x = layers.Activation("swish")(x)

    z_mean = layers.Conv2D(latent_dim, 3, padding="same")(x)
    z_log_var = layers.Conv2D(latent_dim, 3, padding="same")(x)

    def sampling(args):
        z_mean, z_log_var = args
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    z = layers.Lambda(sampling, name="z")([z_mean, z_log_var])
    encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

    # Decoder
    latent_inputs = layers.Input(shape=(img_height // 8, img_width // 8, latent_dim))

    x = layers.Conv2D(512, 3, padding="same")(latent_inputs)
    x = ResBlock(512)(x)
    x = AttentionBlock(512)(x)
    x = ResBlock(512)(x)

    # Upsample 1: 32 -> 64
    x = layers.Conv2DTranspose(512, 3, strides=2, padding="same")(x)
    x = ResBlock(512)(x)
    x = ResBlock(512)(x)

    # Upsample 2: 64 -> 128
    x = layers.Conv2DTranspose(256, 3, strides=2, padding="same")(x)
    x = ResBlock(256)(x)
    x = ResBlock(256)(x)

    # Upsample 3: 128 -> 256
    x = layers.Conv2DTranspose(128, 3, strides=2, padding="same")(x)
    x = ResBlock(128)(x)
    x = ResBlock(128)(x)

    x = layers.GroupNormalization(groups=32)(x)
    x = layers.Activation("swish")(x)
    decoder_outputs = layers.Conv2D(img_channels, 3, padding="same", activation="tanh")(x)

    decoder = Model(latent_inputs, decoder_outputs, name="decoder")

    # Full VAE
    vae_outputs = decoder(encoder(encoder_inputs)[2])
    vae = Model(encoder_inputs, vae_outputs, name="vae")

    return vae, encoder, decoder


class VAEModel(Model):
    def __init__(self, vae, encoder, decoder, kl_weight=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.vae = vae
        self.encoder = encoder
        self.decoder = decoder
        self.kl_weight = kl_weight

        # Perceptual Loss model (VGG19)
        vgg = tf.keras.applications.VGG19(include_top=False, weights="imagenet")
        vgg.trainable = False
        # Use block5_conv2 layer for perceptual features
        self.perceptual_model = Model(vgg.input, vgg.get_layer("block5_conv2").output)

        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")
        self.perceptual_loss_tracker = tf.keras.metrics.Mean(name="perceptual_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
            self.perceptual_loss_tracker
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)

            # Reconstruction Loss (MSE)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.keras.losses.mean_squared_error(data, reconstruction), axis=(1, 2)
                )
            )

            # Perceptual Loss
            # VGG expects [0, 255] or preprocessed images.
            # Our data is likely [-1, 1] due to tanh in decoder.
            p_data = tf.keras.applications.vgg19.preprocess_input((data + 1.0) * 127.5)
            p_reconstruction = tf.keras.applications.vgg19.preprocess_input((reconstruction + 1.0) * 127.5)

            real_features = self.perceptual_model(p_data)
            recon_features = self.perceptual_model(p_reconstruction)
            perceptual_loss = tf.reduce_mean(tf.square(real_features - recon_features))

            # KL Loss
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=(1, 2, 3)))

            total_loss = reconstruction_loss + perceptual_loss * 0.1 + kl_loss * self.kl_weight

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.perceptual_loss_tracker.update_state(perceptual_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "perceptual_loss": self.perceptual_loss_tracker.result(),
        }
