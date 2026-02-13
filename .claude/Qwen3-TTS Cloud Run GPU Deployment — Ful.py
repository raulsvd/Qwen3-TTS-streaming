 Qwen3-TTS Cloud Run GPU Deployment — Full Standalone Plan

 Purpose: This plan is self-contained. Copy it into a Claude instance working on the Terragrunt infra repo to implement the full deployment.
 The infra repo already has Cloud Run Terragrunt modules that can be referenced.

 ---
 1. Background & Goal

 We have a working Qwen3-TTS streaming server — a FastAPI app that:
 - Loads Qwen/Qwen3-TTS-12Hz-1.7B-Base (~3.4GB bfloat16, needs L4 GPU)
 - Uses dffdeeq/Qwen3-TTS-streaming fork for real chunked audio streaming
 - Pre-loads a .pt voice clone prompt file at startup
 - Runs torch.compile + CUDA graphs optimization with a warmup pass
 - Streams WAV audio chunks (PCM int16) as they're generated
 - Exposes POST /v1/audio/speech (streaming + non-streaming) and GET /health

 Current deployment: Vertex AI Workbench on GCP (always-on L4), costs ~$480-720/month.
 Target: Cloud Run with GPU (L4), scale-to-zero, ~$10-100/month depending on usage.

 GCP Project: <PROJECT_ID> (replace throughout)
 Region: us-central1

 ---
 2. Architecture

 [React Frontend]  --->  [Cloud Run Service: qwen-tts]
   (local dev or           Region: us-central1
    Vercel/CF Pages)       GPU: 1x NVIDIA L4
                           Memory: 16Gi, CPU: 4
                           Min instances: 0 (scale-to-zero)
                           Max instances: 1
                           Port: 8080
                           Image from: Artifact Registry

 Cold start ~15-20s (model load + torch.compile warmup). After warm, TTFB ~1.2s, RTF ~0.46x.

 ---
 3. Terraform/Terragrunt Resources Needed

 You already have Cloud Run modules in the infra repo. The resources to create:

 3a. Artifact Registry Repository

 resource "google_artifact_registry_repository" "qwen_tts" {
   location      = "us-central1"
   repository_id = "qwen-tts"
   format        = "DOCKER"
   description   = "Qwen3-TTS streaming server images"
 }

 3b. Cloud Run Service (v2)

 resource "google_cloud_run_v2_service" "qwen_tts" {
   name     = "qwen-tts"
   location = "us-central1"

   template {
     scaling {
       min_instance_count = 0
       max_instance_count = 1
     }

     timeout = "300s"

     containers {
       image = "us-central1-docker.pkg.dev/<PROJECT_ID>/qwen-tts/server:latest"

       ports {
         container_port = 8080
       }

       resources {
         limits = {
           cpu    = "4"
           memory = "16Gi"
           "nvidia.com/gpu" = "1"
         }
         cpu_idle          = false    # --no-cpu-throttling
         startup_cpu_boost = true
       }

       startup_probe {
         http_get {
           path = "/health"
           port = 8080
         }
         initial_delay_seconds = 10
         period_seconds        = 5
         timeout_seconds       = 3
         failure_threshold     = 20   # 20 * 5s = 100s max startup time
       }

       liveness_probe {
         http_get {
           path = "/health"
           port = 8080
         }
         period_seconds    = 30
         timeout_seconds   = 5
         failure_threshold = 3
       }
     }

     node_selector {
       accelerator = "nvidia-l4"
     }
   }

   # Allow unauthenticated access (public API)
   # Remove this block if you want IAM-authenticated access instead
   lifecycle {
     ignore_changes = [
       template[0].containers[0].image,  # managed by CI/CD
     ]
   }
 }

 # Allow unauthenticated invocations
 resource "google_cloud_run_v2_service_iam_member" "public" {
   project  = google_cloud_run_v2_service.qwen_tts.project
   location = google_cloud_run_v2_service.qwen_tts.location
   name     = google_cloud_run_v2_service.qwen_tts.name
   role     = "roles/run.invoker"
   member   = "allUsers"
 }

 3c. Required APIs (enable if not already)

 resource "google_project_service" "apis" {
   for_each = toset([
     "run.googleapis.com",
     "artifactregistry.googleapis.com",
     "cloudbuild.googleapis.com",
   ])
   service            = each.value
   disable_on_destroy = false
 }

 ---
 4. Dockerfile

 Create this in the app repo at server/Dockerfile:

 # --- Stage 1: Build dependencies ---
 FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder

 ENV DEBIAN_FRONTEND=noninteractive
 RUN apt-get update && apt-get install -y \
     python3.12 python3.12-venv python3.12-dev python3-pip \
     git libsndfile1 \
     && rm -rf /var/lib/apt/lists/*

 RUN python3.12 -m pip install uv

 WORKDIR /app
 COPY pyproject.toml .

 # Install all deps into a venv
 RUN python3.12 -m uv venv /app/.venv \
     && . /app/.venv/bin/activate \
     && uv pip install --no-deps "qwen-tts @ git+https://github.com/dffdeeq/Qwen3-TTS-streaming.git" \
     && uv pip install fastapi uvicorn soundfile numpy torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

 # Pre-download model weights (baked into image, ~3.4GB)
 RUN . /app/.venv/bin/activate \
     && python3.12 -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-TTS-12Hz-1.7B-Base')"

 # --- Stage 2: Runtime ---
 FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

 ENV DEBIAN_FRONTEND=noninteractive
 RUN apt-get update && apt-get install -y \
     python3.12 libsndfile1 \
     && rm -rf /var/lib/apt/lists/*

 WORKDIR /app

 # Copy venv and HF cache from builder
 COPY --from=builder /app/.venv /app/.venv
 COPY --from=builder /root/.cache/huggingface /root/.cache/huggingface

 # Copy server code
 COPY tts_server.py .

 # Copy voice prompt (you'll need to place this in server/ before building)
 COPY voice_clone_prompt.pt .

 ENV PATH="/app/.venv/bin:$PATH"
 ENV PORT=8080

 EXPOSE 8080

 # --optimize enables torch.compile + CUDA graphs + warmup run at startup
 CMD ["python3.12", "tts_server.py", \
      "--voice", "voice_clone_prompt.pt", \
      "--port", "8080", \
      "--optimize"]

 Create server/.dockerignore:
 __pycache__
 *.pyc
 .venv
 .git
 uv.lock

 ---
 5. Server Code Changes

 The existing server/tts_server.py needs one small change. In the main() function, replace:

 parser.add_argument("--port", type=int, default=8081)

 with:

 parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8080)))

 And add import os at the top (alongside the existing imports).

 Everything else (streaming endpoint, health check, torch.compile, warmup) works as-is on Cloud Run.

 ---
 6. Full Server Code Reference

 For context, here's the complete tts_server.py (so the infra Claude can understand what's being deployed):

 """
 Streaming FastAPI server for Qwen3-TTS with pre-loaded .pt voice prompt.
 Uses dffdeeq/Qwen3-TTS-streaming fork for real chunked audio streaming.
 """

 import argparse, io, os, struct, time
 import numpy as np
 import soundfile as sf
 import torch
 from fastapi import FastAPI, HTTPException
 from fastapi.responses import Response, StreamingResponse
 from pydantic import BaseModel

 app = FastAPI()
 tts_model = None
 voice_prompt = None

 class GenerateRequest(BaseModel):
     text: str
     language: str = "Auto"
     max_new_tokens: int = 2048
     temperature: float = 0.9
     stream: bool = True
     emit_every_frames: int = 8
     decode_window_frames: int = 80

 @app.post("/v1/audio/speech")
 async def generate_speech(req: GenerateRequest):
     if tts_model is None or voice_prompt is None:
         raise HTTPException(status_code=503, detail="Model not loaded")
     if req.stream:
         return StreamingResponse(stream_audio(req), media_type="audio/wav",
                                  headers={"X-Streaming": "true"})
     else:
         return await generate_full(req)

 async def stream_audio(req: GenerateRequest):
     """Yields WAV header + PCM chunks. Must be async def because torch.compile
     + CUDA graphs use thread-local storage (must run on main event loop thread)."""
     t0 = time.perf_counter()
     first_chunk = True
     chunk_count = 0
     total_samples = 0
     sr = 24000

     for pcm_chunk, sr in tts_model.stream_generate_voice_clone(
         text=req.text, language=req.language, voice_clone_prompt=voice_prompt,
         emit_every_frames=req.emit_every_frames,
         decode_window_frames=req.decode_window_frames,
         max_new_tokens=req.max_new_tokens, temperature=req.temperature,
     ):
         if first_chunk:
             print(f"TTFB: {(time.perf_counter()-t0)*1000:.0f}ms")
             yield make_wav_header(sr)
             first_chunk = False
         yield pcm_to_int16_bytes(pcm_chunk)
         chunk_count += 1
         total_samples += len(pcm_chunk)

     total_time = (time.perf_counter() - t0) * 1000
     if chunk_count > 0:
         dur = total_samples / sr
         print(f"Streamed {chunk_count} chunks, {dur:.2f}s audio in {total_time:.0f}ms (RTF: {total_time/1000/dur:.2f}x)")

 @app.get("/health")
 async def health():
     return {"status": "ok", "voice_loaded": voice_prompt is not None, "streaming": True}

 Key behaviors:
 - stream_audio is async def (not sync) — CUDA graphs require same thread as warmup
 - WAV header uses 0xFFFFFFFF for unknown length (streaming)
 - PCM is int16, sample rate parsed from model output (typically 24000 Hz)
 - --optimize flag triggers torch.compile + CUDA graphs + warmup at startup (adds ~2-5min to cold start)

 ---
 7. Build & Deploy Sequence

 # 1. Copy voice prompt file into server/ directory
 cp ~/voice_clone_prompt_as62p9fg.pt server/voice_clone_prompt.pt

 # 2. Build with Cloud Build (handles large images, no local GPU needed)
 gcloud builds submit server/ \
   --project <PROJECT_ID> \
   --tag us-central1-docker.pkg.dev/<PROJECT_ID>/qwen-tts/server:latest \
   --machine-type=e2-highcpu-8 \
   --timeout=45m

 # 3. Apply Terragrunt (creates Artifact Registry + Cloud Run service)
 cd <infra-repo>/path/to/qwen-tts/
 terragrunt apply

 # 4. Verify
 curl https://qwen-tts-HASH-uc.a.run.app/health
 # Should return: {"status":"ok","voice_loaded":true,"streaming":true}

 ---
 8. Terragrunt Module Structure

 Suggested layout in the infra repo (adapt to your existing conventions):

 <infra-repo>/
 └── <env>/
     └── qwen-tts/
         ├── terragrunt.hcl      # References existing cloud_run module
         ├── artifact-registry/
         │   └── terragrunt.hcl  # Artifact Registry repo
         └── cloud-run/
             └── terragrunt.hcl  # Cloud Run service with GPU config

 Since you already have Cloud Run modules, the main new things are:
 - The node_selector.accelerator = "nvidia-l4" for GPU
 - The resources.limits["nvidia.com/gpu"] = "1" on the container
 - The cpu_idle = false (must not throttle CPU with GPU workloads)
 - The startup probe with high failure_threshold (model load takes time)

 ---
 9. Cost Summary
 ┌──────────────────────────────────────────────┬──────────────┐
 │                   Scenario                   │ Monthly Cost │
 ├──────────────────────────────────────────────┼──────────────┤
 │ Current (Vertex AI Workbench, always-on L4)  │ $480-720     │
 ├──────────────────────────────────────────────┼──────────────┤
 │ Cloud Run GPU, scale-to-zero, ~30 hrs/month  │ ~$20         │
 ├──────────────────────────────────────────────┼──────────────┤
 │ Cloud Run GPU, scale-to-zero, ~200 hrs/month │ ~$134        │
 ├──────────────────────────────────────────────┼──────────────┤
 │ Cloud Run GPU, min-instances=1 (always warm) │ ~$480        │
 └──────────────────────────────────────────────┴──────────────┘
 L4 GPU on Cloud Run: $0.67/hour. CPU/memory charges are additional but minimal.

 ---
 10. Notes & Gotchas

 1. Cold start: ~15-20s for model load + torch.compile warmup. The startup probe handles this gracefully — Cloud Run won't route traffic
 until /health returns 200.
 2. Image size: The Docker image will be ~8-10GB (CUDA runtime + model weights). Cloud Build handles this fine. Consider using --disk-size if
  the default 100GB isn't enough.
 3. Request timeout: Set to 300s (5 min). Long texts can take a while to generate. Streaming means the client sees audio before the 5 min is
 up.
 4. CUDA graphs + threading: The stream_audio function MUST be async def (not sync def). Starlette runs sync generators in a threadpool, but
 CUDA graphs store state in thread-local storage. Using async keeps it on the main event loop thread (same as warmup).
 5. Voice prompt: The .pt file must be copied into server/ before building the Docker image. It contains the speaker embedding for voice
 cloning.
 6. Model variant: Qwen/Qwen3-TTS-12Hz-1.7B-Base — this is the base model that supports voice cloning. Do NOT use the instruct variant.
 7. Frontend: The React app at qwen-mvp/ connects via Vite proxy. Update vite.config.ts proxy target to the Cloud Run URL:
 target: process.env.VITE_API_URL || 'http://localhost:8081'
 7. Then run: VITE_API_URL=https://qwen-tts-HASH-uc.a.run.app npm run dev