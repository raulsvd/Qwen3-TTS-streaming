"""
Test voice cloning prosody variations.

Systematically tests different parameter combinations to find settings
that produce more natural, expressive (less flat) cloned speech.
Supports sentence-by-sentence generation with configurable silence
between sentences for more natural pacing.

Usage:
    uv run python examples/test_prosody_variations.py --voice tom_hegna
    uv run python examples/test_prosody_variations.py --voice tom_hegna_2 --pause_ms 500
"""

import argparse
import os
import re
import time
import json
import numpy as np
import torch
import soundfile as sf

from qwen_tts import Qwen3TTSModel


def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)


def split_into_sentences(text: str) -> list[str]:
    """Split text into sentences at sentence-ending punctuation."""
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in parts if s.strip()]


def generate_with_pauses(
    tts: Qwen3TTSModel,
    text: str,
    language: str,
    voice_clone_prompt,
    pause_ms: int,
    gen_kwargs: dict,
) -> tuple[np.ndarray, int]:
    """Generate audio with silence inserted between sentences."""
    sentences = split_into_sentences(text)
    chunks = []
    sr = 24000  # will be overwritten

    for j, sentence in enumerate(sentences):
        wavs, sr = tts.generate_voice_clone(
            text=sentence,
            language=language,
            voice_clone_prompt=voice_clone_prompt,
            **gen_kwargs,
        )
        chunks.append(wavs[0])
        if j < len(sentences) - 1:
            silence = np.zeros(int(sr * pause_ms / 1000), dtype=np.float32)
            chunks.append(silence)

    return np.concatenate(chunks), sr


def main():
    parser = argparse.ArgumentParser(description="Test voice cloning prosody variations")
    parser.add_argument(
        "--voice", default="tom_hegna",
        help="Voice name. Maps to assets/<voice>_voice.mp3 and assets/<voice>_transcript.txt",
    )
    parser.add_argument(
        "--pause_ms", type=int, default=0,
        help="Silence between sentences in ms. 0 = generate full text at once (default: 0)",
    )
    args = parser.parse_args()

    voice = args.voice
    pause_ms = args.pause_ms
    ref_audio_path = f"assets/{voice}_voice.mp3"
    ref_transcript_path = f"assets/{voice}_transcript.txt"

    if not os.path.exists(ref_audio_path):
        raise FileNotFoundError(f"Reference audio not found: {ref_audio_path}")
    if not os.path.exists(ref_transcript_path):
        raise FileNotFoundError(f"Reference transcript not found: {ref_transcript_path}")

    device = "cuda:0"
    MODEL_PATH = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    OUT_DIR = "prosody_test_output"
    ensure_dir(OUT_DIR)

    # --- Reference audio & text ---
    with open(ref_transcript_path, "r") as f:
        ref_text = f.read().strip()

    with open("assets/reference_output.txt", "r") as f:
        syn_text = f.read().strip()
    syn_lang = "Auto"

    # --- Load model ---
    print("=" * 70)
    print(f"Loading model... (voice: {voice}, pause_ms: {pause_ms})")
    print("=" * 70)
    t0 = time.time()
    tts = Qwen3TTSModel.from_pretrained(
        MODEL_PATH,
        device_map=device,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    print(f"Model loaded in {time.time() - t0:.1f}s")

    # --- Build voice clone prompt (ICL mode only) ---
    print("\nCreating voice clone prompt (ICL mode)...")
    t0 = time.time()
    prompt_icl = tts.create_voice_clone_prompt(
        ref_audio=ref_audio_path,
        ref_text=ref_text,
        x_vector_only_mode=False,
    )
    print(f"  ICL prompt created in {time.time() - t0:.1f}s")

    if pause_ms > 0:
        sentences = split_into_sentences(syn_text)
        print(f"\nSentence-by-sentence mode: {len(sentences)} sentences, {pause_ms}ms pause between each")
        for i, s in enumerate(sentences, 1):
            print(f"  {i}. {s[:70]}{'...' if len(s) > 70 else ''}")

    # --- Baseline parameters ---
    baseline = dict(
        do_sample=True,
        top_k=50,
        top_p=1.0,
        temperature=0.9,
        repetition_penalty=1.05,
        subtalker_dosample=True,
        subtalker_top_k=50,
        subtalker_top_p=1.0,
        subtalker_temperature=0.9,
        max_new_tokens=2048,
    )

    # --- 4 focused variations ---
    variations = [
        ("baseline_icl", {**baseline}),
        ("top_k_100", {**baseline, "top_k": 100}),
        ("natural_flow", {**baseline, "temperature": 1.0, "top_k": 100,
                          "repetition_penalty": 1.0, "subtalker_temperature": 1.0}),
        ("expressive", {**baseline, "temperature": 1.1, "top_k": 100, "top_p": 0.95,
                        "repetition_penalty": 1.0, "subtalker_temperature": 1.1}),
    ]

    # --- Run all variations ---
    results = []
    total_start = time.time()

    print(f"\n{'=' * 70}")
    print(f"Voice: {voice} | Pause: {pause_ms}ms")
    print(f"Running {len(variations)} variations")
    print(f"Synthesis text: {syn_text[:80]}...")
    print(f"{'=' * 70}\n")

    for i, (name, gen_kwargs) in enumerate(variations, 1):
        full_name = f"{voice}_{name}"
        print(f"[{i}/{len(variations)}] {full_name}")
        torch.cuda.synchronize()
        t0 = time.time()

        if pause_ms > 0:
            audio, sr = generate_with_pauses(
                tts, syn_text, syn_lang, prompt_icl, pause_ms, gen_kwargs,
            )
        else:
            wavs, sr = tts.generate_voice_clone(
                text=syn_text,
                language=syn_lang,
                voice_clone_prompt=prompt_icl,
                **gen_kwargs,
            )
            audio = wavs[0]

        torch.cuda.synchronize()
        elapsed = time.time() - t0
        audio_duration = len(audio) / sr

        # Save audio
        wav_path = os.path.join(OUT_DIR, f"{full_name}.wav")
        sf.write(wav_path, audio, sr)

        # Save params
        params_path = os.path.join(OUT_DIR, f"{full_name}_params.txt")
        params_info = {
            "voice": voice,
            "mode": "ICL",
            "pause_ms": pause_ms,
            **{k: v for k, v in gen_kwargs.items() if k != "max_new_tokens"},
        }
        with open(params_path, "w") as f:
            json.dump(params_info, f, indent=2)

        rtf = elapsed / audio_duration if audio_duration > 0 else 0
        print(f"  -> {elapsed:.1f}s gen, {audio_duration:.1f}s audio, RTF={rtf:.2f}")

        results.append({
            "name": full_name,
            "gen_time": elapsed,
            "audio_duration": audio_duration,
            "rtf": rtf,
            "params": params_info,
        })

    # --- Summary table ---
    print(f"\n{'=' * 95}")
    print(f"SUMMARY (voice: {voice}, pause: {pause_ms}ms)")
    print(f"{'=' * 95}")
    print(f"{'Name':<35} {'Temp':>5} {'SubTemp':>8} {'RepPen':>7} {'TopK':>5} {'TopP':>5} {'GenTime':>8} {'Audio':>7} {'RTF':>5}")
    print("-" * 95)

    for r in results:
        p = r["params"]
        print(
            f"{r['name']:<35} "
            f"{p.get('temperature', '-'):>5} "
            f"{p.get('subtalker_temperature', '-'):>8} "
            f"{p.get('repetition_penalty', '-'):>7} "
            f"{p.get('top_k', '-'):>5} "
            f"{p.get('top_p', '-'):>5} "
            f"{r['gen_time']:>7.1f}s "
            f"{r['audio_duration']:>6.1f}s "
            f"{r['rtf']:>5.2f}"
        )

    total_time = time.time() - total_start
    print(f"\nTotal time: {total_time:.1f}s")
    print(f"Output directory: {OUT_DIR}/")
    print(f"\nListen to the files and compare expressiveness!")
    print(f"Key things to listen for:")
    print(f"  - Pitch variation (monotone vs dynamic)")
    print(f"  - Natural pauses and rhythm")
    print(f"  - Emphasis on important words")
    print(f"  - Overall naturalness vs robotic quality")


if __name__ == "__main__":
    main()
