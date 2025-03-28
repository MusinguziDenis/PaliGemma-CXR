"""Configure model and processor."""

import functools
import re
from pathlib import Path

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from utils import get_device

device = get_device()

model_id = "google/paligemma-3b-pt-224"
num_seg_tokens = 16

processor = AutoProcessor.from_pretrained(model_id)


def get_model(config: dict) -> PaliGemmaForConditionalGeneration:
    """Get the model.

    Args:
        config (dict): Configuration dictionary.

    Returns:
        PaliGemmaForConditionalGeneration: Model.

    """
    model = PaliGemmaForConditionalGeneration.from_pretrained(
                                    config["model_id"],
                                    torch_dtype=config["model_dtype"],
                                    device_map=device,
                                    revision=config["model_revision"],
                        )
    for param in model.vision_tower.parameters():
        param.requires_grad = False

    for param in model.multi_modal_projector.parameters():
        param.requires_grad = False

    for param in model.language_model.parameters():
        param.requires_grad = True

    return model



# Mask reconstruction
_MODEL_PATH = "vae-oid.npz"

_SEGMENT_DETECT_RE = re.compile(
    r"(.*?)" +
    r"<loc(\d{4})>" * 4 + r"\s*" +
    "(?:%s)?" % (r"<seg(\d{3})>" * 16) +
    r"\s*([^;<>]+)? ?(?:; )?",
)

def _get_params(checkpoint: str) -> dict[str, np.ndarray]:
    """Convert PyTorch checkpoint to Flax params.

    Args:
        checkpoint (str): Path to the checkpoint.

    Returns:
        dict[str, np.ndarray]: Dictionary containing the parameters.

    """

    def transp(kernel: np.ndarray) -> np.ndarray:
        """Tranpose the kernel from PyTorch to Flax format.

        Args:
            kernel (np.ndarray): Kernel to transpose.

        Returns:
            np.ndarray: Transposed kernel.

        """
        return np.transpose(kernel, (2, 3, 1, 0))

    def conv(name: str) -> dict[str, np.ndarray]:
        """Get the convolutional layer parameters.

        Args:
            name (str): Name of the layer.

        Returns:
            dict[str, np.ndarray]: Dictionary containing the bias and kernel.

        """
        return {
            "bias": checkpoint[name + ".bias"],
            "kernel": transp(checkpoint[name + ".weight"]),
        }

    def resblock(name: str) -> dict[str, np.ndarray]:
        """Get the residual block parameters.

        Args:
            name (str): Name of the layer.

        Returns:
            dict[str, np.ndarray]: Dictionary containing the bias and kernel.

        """
        return {
            "Conv_0": conv(name + ".0"),
            "Conv_1": conv(name + ".2"),
            "Conv_2": conv(name + ".4"),
        }

    return {
        "_embeddings": checkpoint["_vq_vae._embedding"],
        "Conv_0": conv("decoder.0"),
        "ResBlock_0": resblock("decoder.2.net"),
        "ResBlock_1": resblock("decoder.3.net"),
        "ConvTranspose_0": conv("decoder.4"),
        "ConvTranspose_1": conv("decoder.6"),
        "ConvTranspose_2": conv("decoder.8"),
        "ConvTranspose_3": conv("decoder.10"),
        "Conv_1": conv("decoder.12"),
    }

def _quantized_values_from_codebook_indices(
        codebook_indices: jnp.ndarray,
        embeddings: jnp.ndarray,
        ) -> jnp.ndarray:
    """Convert codebook indices to quantized values.

    Args:
        codebook_indices (jnp.ndarray): Codebook indices of shape [B, 16].
        embeddings (jnp.ndarray): Embeddings of shape [128, 128].

    Returns:
        jnp.ndarray: Quantized values of shape [B, 4, 4, 128].

    """
    batch_size, num_tokens = codebook_indices.shape
    if num_tokens != num_seg_tokens:
        error_msg = f"Expected 16 tokens, but got {num_tokens}.\
                          Shape: {codebook_indices.shape}"
        raise ValueError(error_msg)

    unused_num_embeddings, embedding_dim = embeddings.shape

    encodings = jnp.take(embeddings, codebook_indices.reshape(-1), axis=0)
    return encodings.reshape((batch_size, 4, 4, embedding_dim))


@functools.cache
def _get_reconstruct_masks() -> jax.core.Jaxpr:
    """Reconstructs masks from codebook indices.

    Returns:
        A function that expects indices shaped `[B, 16]` of dtype int32,
        eachranging from 0 to 127 (inclusive), and that returns a decoded
        masks sized`[B, 64, 64, 1]`, of dtype float32, in range [-1, 1].

    """

    class ResBlock(nn.Module):
        """Residual block for the decoder."""

        features: int

        @nn.compact
        def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
            """Apply the residual block to the input.

            Args:
                x (jnp.ndarray): Input tensor of shape [B, H, W, C].

            Returns:
                jnp.ndarray: Output tensor of shape [B, H, W, C].

            """
            original_x = x
            x = nn.Conv(features=self.features,
                        kernel_size=(3, 3),
                        padding=1)(x)
            x = nn.relu(x)
            x = nn.Conv(features=self.features,
                        kernel_size=(3, 3),
                        padding=1)(x)
            x = nn.relu(x)
            x = nn.Conv(features=self.features,
                        kernel_size=(1, 1),
                        padding=0)(x)
            return x + original_x

    class Decoder(nn.Module):
        """Upscales quantized vectors to mask."""

        @nn.compact
        def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
            """Apply the decoder to the input.

            Args:
                x (jnp.ndarray): Input tensor of shape [B, 4, 4, 128].

            Returns:
                jnp.ndarray: Output tensor of shape [B, 64, 64, 1].

            """
            num_res_blocks = 2
            dim = 128
            num_upsample_layers = 4

            x = nn.Conv(features=dim, kernel_size=(1, 1), padding=0)(x)
            x = nn.relu(x)

            for _ in range(num_res_blocks):
                x = ResBlock(features=dim)(x)

            for _ in range(num_upsample_layers):
                x = nn.ConvTranspose(
                    features=dim,
                    kernel_size=(4, 4),
                    strides=(2, 2),
                    padding=2,
                    transpose_kernel=True,
                )(x)
                x = nn.relu(x)
                dim //= 2

            return nn.Conv(features=1, kernel_size=(1, 1), padding=0)(x)

    def reconstruct_masks(
            codebook_indices: jnp.ndarray,
            )-> jnp.ndarray:
        """Reconstructs masks from codebook indices.

        Args:
            codebook_indices (jnp.ndarray): Codebook indices of shape [B, 16].

        Returns:
            jnp.ndarray: Reconstructed masks of shape [B, 64, 64, 1].

        """
        quantized = _quantized_values_from_codebook_indices(
            codebook_indices, params["_embeddings"],
        )
        return Decoder().apply({"params": params}, quantized)

    param_path = Path(_MODEL_PATH)
    with param_path.open("rb") as f:
        params = _get_params(dict(np.load(f)))

    return jax.jit(reconstruct_masks, backend="cpu")
