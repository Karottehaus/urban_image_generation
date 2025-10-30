import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.losses import MeanSquaredError
from PIL import Image
import os
from glob import glob
from settings import IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, MERGED_IMG_CHANNELS
from settings import BATCH_SIZE, EPOCHS, NUM_TIMESTEPS, BETA_START, BETA_END, LEARNING_RATE, NUM_SAMPLES


def load_paired_images(left_folder: str, right_folder: str):
    left_paths = sorted(glob(os.path.join(left_folder, '*.png')))
    right_paths = sorted(glob(os.path.join(right_folder, '*.png')))

    merged_images = []

    for left_path, right_path in zip(left_paths, right_paths):
        left_img = np.array(Image.open(left_path).convert('RGB')) / 127.5 - 1.0
        right_img = np.array(Image.open(right_path).convert('RGB')) / 127.5 - 1.0
        # Concatenate along channel dimension → shape (H, W, 6)
        merged = np.concatenate([left_img, right_img], axis=-1)
        merged_images.append(merged)

    return np.array(merged_images, dtype=np.float32)


def build_noise_predictor(img_height, img_width, merged_img_channels):
    """Build U-Net style noise prediction network for images"""

    # Input layers
    img_input = layers.Input(shape=(img_height, img_width, merged_img_channels), name="noisy_image")
    timestep_input = layers.Input(shape=(1,), name="timestep")

    # Timestep embedding
    t_embed = layers.Dense(256, activation="swish")(timestep_input)
    t_embed = layers.Dense(256, activation="swish")(t_embed)

    # Encoder (downsampling)
    # Block 1
    x1 = layers.Conv2D(64, 3, padding="same", activation="relu")(img_input)
    x1 = layers.Conv2D(64, 3, padding="same", activation="relu")(x1)

    # Add timestep embedding via FiLM (Feature-wise Linear Modulation)
    t_embed_1 = layers.Dense(64)(t_embed)
    t_embed_1 = layers.Reshape((1, 1, 64))(t_embed_1)
    x1 = x1 * t_embed_1

    pool1 = layers.MaxPooling2D(2)(x1)

    # Block 2
    x2 = layers.Conv2D(128, 3, padding="same", activation="relu")(pool1)
    x2 = layers.Conv2D(128, 3, padding="same", activation="relu")(x2)

    t_embed_2 = layers.Dense(128)(t_embed)
    t_embed_2 = layers.Reshape((1, 1, 128))(t_embed_2)
    x2 = x2 * t_embed_2

    pool2 = layers.MaxPooling2D(2)(x2)

    # Block 3
    x3 = layers.Conv2D(256, 3, padding="same", activation="relu")(pool2)
    x3 = layers.Conv2D(256, 3, padding="same", activation="relu")(x3)

    t_embed_3 = layers.Dense(256)(t_embed)
    t_embed_3 = layers.Reshape((1, 1, 256))(t_embed_3)
    x3 = x3 * t_embed_3

    pool3 = layers.MaxPooling2D(2)(x3)

    # Block 4
    x4 = layers.Conv2D(512, 3, padding="same", activation="relu")(pool3)
    x4 = layers.Conv2D(512, 3, padding="same", activation="relu")(x4)

    t_embed_4 = layers.Dense(512)(t_embed)
    t_embed_4 = layers.Reshape((1, 1, 512))(t_embed_4)
    x4 = x4 * t_embed_4

    pool4 = layers.MaxPooling2D(2)(x4)

    # Bottleneck
    bottleneck = layers.Conv2D(1024, 3, padding="same", activation="relu")(pool4)
    bottleneck = layers.Conv2D(1024, 3, padding="same", activation="relu")(bottleneck)

    t_embed_b = layers.Dense(1024)(t_embed)
    t_embed_b = layers.Reshape((1, 1, 1024))(t_embed_b)
    bottleneck = bottleneck * t_embed_b

    # Decoder (upsampling)
    # Block 4
    up4 = layers.UpSampling2D(2)(bottleneck)
    up4 = layers.Concatenate()([up4, x4])
    up4 = layers.Conv2D(512, 3, padding="same", activation="relu")(up4)
    up4 = layers.Conv2D(512, 3, padding="same", activation="relu")(up4)

    # Block 3
    up3 = layers.UpSampling2D(2)(up4)
    up3 = layers.Concatenate()([up3, x3])
    up3 = layers.Conv2D(256, 3, padding="same", activation="relu")(up3)
    up3 = layers.Conv2D(256, 3, padding="same", activation="relu")(up3)

    # Block 2
    up2 = layers.UpSampling2D(2)(up3)
    up2 = layers.Concatenate()([up2, x2])
    up2 = layers.Conv2D(128, 3, padding="same", activation="relu")(up2)
    up2 = layers.Conv2D(128, 3, padding="same", activation="relu")(up2)

    # Block 1
    up1 = layers.UpSampling2D(2)(up2)
    up1 = layers.Concatenate()([up1, x1])
    up1 = layers.Conv2D(64, 3, padding="same", activation="relu")(up1)
    up1 = layers.Conv2D(64, 3, padding="same", activation="relu")(up1)

    # Output
    predicted_noise = layers.Conv2D(merged_img_channels, 1, padding="same")(up1)
    return Model([img_input, timestep_input], predicted_noise, name="noise_predictor")


class ImageDiffusionModel(Model):
    def __init__(self, noise_predictor: Model, num_timesteps: int,
                 beta_start: float, beta_end: float, **kwargs):
        super().__init__(**kwargs)
        self.noise_predictor = noise_predictor
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end

        # Linear beta schedule
        self.betas = tf.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = tf.math.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = tf.concat([[1.0], self.alphas_cumprod[:-1]], axis=0)

        # Pre-calculate values for sampling
        self.sqrt_alphas_cumprod = tf.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = tf.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = tf.sqrt(1.0 / self.alphas)
        self.posterior_variance = (self.betas * (1.0 - self.alphas_cumprod_prev) /
                                   (1.0 - self.alphas_cumprod))

        self.mse_loss_fn = MeanSquaredError()

    def compile(self, optimizer):
        super().compile()
        self.optimizer = optimizer

    def q_sample(self, x_start: tf.Tensor, t: tf.Tensor, epsilon: tf.Tensor) -> tf.Tensor:
        """Forward diffusion process: add noise to images"""
        sqrt_alpha_cumprod_t = tf.gather(self.sqrt_alphas_cumprod, t)
        sqrt_one_minus_alpha_cumprod_t = tf.gather(self.sqrt_one_minus_alphas_cumprod, t)

        # Reshape for broadcasting
        sqrt_alpha_cumprod_t = tf.reshape(sqrt_alpha_cumprod_t, [-1, 1, 1, 1])
        sqrt_one_minus_alpha_cumprod_t = tf.reshape(sqrt_one_minus_alpha_cumprod_t, [-1, 1, 1, 1])

        x_t = sqrt_alpha_cumprod_t * x_start + sqrt_one_minus_alpha_cumprod_t * epsilon
        return x_t

    def train_step(self, images: tf.Tensor) -> dict:
        """Training step for diffusion model"""
        batch_size = tf.shape(images)[0]

        with tf.GradientTape() as tape:
            t = tf.random.uniform((batch_size,), minval=0, maxval=self.num_timesteps, dtype=tf.int32)

            epsilon = tf.random.normal(shape=tf.shape(images))

            x_t = self.q_sample(images, t, epsilon)

            predicted_noise = self.noise_predictor([x_t, tf.cast(t, tf.float32)])

            loss = self.mse_loss_fn(epsilon, predicted_noise)

        grads = tape.gradient(loss, self.noise_predictor.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.noise_predictor.trainable_weights))

        return {"loss": loss}


def train_diffusion(images: np.ndarray, epochs: int, batch_size: int) -> ImageDiffusionModel:
    noise_predictor = build_noise_predictor(IMG_HEIGHT, IMG_WIDTH, MERGED_IMG_CHANNELS)

    diffusion_model = ImageDiffusionModel(
        noise_predictor,
        num_timesteps=NUM_TIMESTEPS,
        beta_start=BETA_START,
        beta_end=BETA_END
    )

    diffusion_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE))

    early_stopping = EarlyStopping(
        monitor="loss",
        patience=50,
        restore_best_weights=True,
        min_delta=1e-6,
        verbose=1
    )

    lr_scheduler = ReduceLROnPlateau(
        monitor="loss",
        factor=0.5,
        patience=20,
        verbose=1,
        min_lr=1e-8
    )

    dataset = tf.data.Dataset.from_tensor_slices(images)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    diffusion_model.fit(
        dataset,
        epochs=epochs,
        callbacks=[early_stopping, lr_scheduler]
    )

    os.makedirs("trained_models", exist_ok=True)
    diffusion_model.noise_predictor.save("trained_models/noise_predictor.h5")
    print("Trained model saved.")


if __name__ == "__main__":
    images = load_paired_images(left_folder="left", right_folder="right")
    train_diffusion(images, epochs=EPOCHS, batch_size=BATCH_SIZE)
