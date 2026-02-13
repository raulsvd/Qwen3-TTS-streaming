# Qwen3-TTS Streaming

Streaming inference implementation for [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) that the official repo doesn't provide.

The official team mentions "Extreme Low-Latency Streaming Generation" in their paper and marketing, but the actual streaming code was never released - they point users to vLLM-Omni, which still doesn't support online serving.

This fork adds real streaming generation directly to the `qwen-tts` package.

In addition to real streaming, this fork includes an **~6x inference speedup** vs upstream qwen-tts - both for non-streaming generation and streaming mode.

## What's Added

- `stream_generate_pcm()` - real-time PCM audio streaming
- `stream_generate_voice_clone()` - streaming with voice cloning

## Benchmark (RTX 5090)

### Non-streaming (full inference)

<img width="602" height="145" alt="image" src="https://github.com/user-attachments/assets/0cbfcc71-e854-46e2-81bc-ec3955ff3ff0" />



### Streaming

<img width="766" height="183" alt="image" src="https://github.com/user-attachments/assets/f5df9a38-e091-47ae-a08f-ef364f8710ea" />



## Usage

See examples/
- [test_streaming_optimized.py](https://github.com/dffdeeq/Qwen3-TTS-streaming/blob/main/examples/test_streaming_optimized.py)
- [test_optimized_no_streaming.py](https://github.com/dffdeeq/Qwen3-TTS-streaming/blob/main/examples/test_optimized_no_streaming.py)

## Installation (python 3.12, Linux)

### 1. Install CUDA Toolkit 12.8 and system dependencies

```bash
# Add the NVIDIA CUDA 12.8 repository
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update

# Install the 12.8 toolkit and system dependencies
sudo apt-get install -y cuda-toolkit-12-8 ninja-build sox libsox-fmt-all ffmpeg

# Add CUDA to your PATH
echo 'export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### 2. Install the package

```bash
git clone https://github.com/dffdeeq/Qwen3-TTS-streaming.git
cd Qwen3-TTS-streaming
uv sync --all-packages --all-extras
```

> **Note:** Running `uv sync` a second time may show `flash-attn` being reinstalled — this is a known uv quirk with local version identifiers in the wheel and is harmless.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `emit_every_frames` | 4 | Emit audio every N frames (~0.33s at 12Hz) |
| `decode_window_frames` | 80 | Decoder context window |

## Why This Exists

From official Qwen3-TTS README:
> Now only offline inference is supported. Online serving will be supported later.

This fork provides streaming now, without waiting for vLLM-Omni updates.

---

Based on [QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)
