import numpy as np
import tensorflow as tf
from PIL import Image
import os
from settings import NUM_TIMESTEPS, BETA_START, BETA_END


class ImageDiffusionModel:
    def __init__(self, noise_predictor, num_timesteps, beta_start, beta_end, latent_dim=4):
        self.noise_predictor = noise_predictor
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.latent_dim = latent_dim

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

    def p_sample(self, x_t: tf.Tensor, control_latent: tf.Tensor, t: int) -> tf.Tensor:
        """Reverse diffusion process: denoise latents"""
        batch_size = tf.shape(x_t)[0]
        t_batch = tf.ones((batch_size,), dtype=tf.int32) * t

        predicted_noise = self.noise_predictor([x_t, control_latent, tf.cast(t_batch, tf.float32)])

        alpha_cumprod_t = tf.gather(self.alphas_cumprod, t)
        beta_t = tf.gather(self.betas, t)
        sqrt_recip_alpha_t = tf.gather(self.sqrt_recip_alphas, t)
        posterior_variance_t = tf.gather(self.posterior_variance, t)

        # Reshape for broadcasting
        alpha_cumprod_t = tf.reshape(alpha_cumprod_t, [1, 1, 1, 1])
        beta_t = tf.reshape(beta_t, [1, 1, 1, 1])
        sqrt_recip_alpha_t = tf.reshape(sqrt_recip_alpha_t, [1, 1, 1, 1])
        posterior_variance_t = tf.reshape(posterior_variance_t, [1, 1, 1, 1])

        model_mean = sqrt_recip_alpha_t * (x_t - beta_t / tf.sqrt(1.0 - alpha_cumprod_t) * predicted_noise)

        if t > 0:
            epsilon = tf.random.normal(shape=tf.shape(x_t))
            return model_mean + tf.sqrt(posterior_variance_t) * epsilon
        else:
            return model_mean

    @tf.function
    def generate(self, control_latents: tf.Tensor) -> tf.Tensor:
        """Generate in latent space"""
        batch_size = tf.shape(control_latents)[0]
        h, w = control_latents.shape[1:3]

        # Initial random noise in latent space
        x = tf.random.normal((batch_size, h, w, self.latent_dim))

        for t in reversed(range(self.num_timesteps)):
            x = self.p_sample(x, control_latents, t)
        return x


def save_generated_images(generated_images: np.ndarray, output_dir: str = "generated_images"):
    """Save generated images to disk"""
    os.makedirs(output_dir, exist_ok=True)

    # Denormalize from [-1, 1] to [0, 255]
    generated_images = np.clip(generated_images, -1.0, 1.0)
    generated_images = ((generated_images + 1.0) * 127.5).astype(np.uint8)

    for i, img in enumerate(generated_images):
        img_pil = Image.fromarray(img)
        img_pil.save(os.path.join(output_dir, f"generated_{i:04d}.png"))

    print(f"Saved {len(generated_images)} images to {output_dir}")


if __name__ == "__main__":
    print("Loading trained models...")
    noise_predictor = tf.keras.models.load_model("trained_models/noise_predictor.keras", compile=False)
    encoder = tf.keras.models.load_model("trained_models/encoder.keras", compile=False)
    decoder = tf.keras.models.load_model("trained_models/decoder.keras", compile=False)

    diffusion_model = ImageDiffusionModel(noise_predictor, NUM_TIMESTEPS, BETA_START, BETA_END)

    img_path = "../right/1218.png"
    img = Image.open(img_path).convert("RGB")
    img = np.array(img) / 127.5 - 1.0
    img = img.astype(np.float32)
    right_tensor = tf.expand_dims(img, axis=0)

    print("Encoding control image...")
    _, _, control_latent = encoder.predict(right_tensor)

    print("Generating in latent space...")
    generated_latent = diffusion_model.generate(control_latent)

    print("Decoding generated latents...")
    generated_images = decoder.predict(generated_latent)

    save_generated_images(generated_images)
