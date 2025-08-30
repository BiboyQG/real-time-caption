#!/usr/bin/env python3
"""
Real-time lecture transcription using Parakeet MLX
Captures audio from microphone and provides live captions
"""

import threading
import time
import queue
import numpy as np
from datetime import datetime
import sounddevice as sd
import mlx.core as mx
from parakeet_mlx import from_pretrained


class LiveTranscriber:
    def __init__(self, model_name="mlx-community/parakeet-tdt-0.6b-v2",
                 sample_rate=16000, chunk_duration=1.0):
        """
        Initialize live transcriber

        Args:
            model_name: Parakeet model to use
            sample_rate: Audio sample rate (16kHz is standard for ASR)
            chunk_duration: Duration of audio chunks in seconds
        """
        print(f"Loading model: {model_name}")
        self.model = from_pretrained(model_name)
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.chunk_size = int(sample_rate * chunk_duration)

        # Audio queue for real-time processing
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.transcription_thread = None

        # Transcription storage
        self.full_transcription = []
        self.current_text = ""

    def audio_callback(self, indata, frames, time, status):
        """Audio callback function for sounddevice"""
        if status:
            print(f"Audio status: {status}")

        # Convert to mono and add to queue
        audio_chunk = indata[:, 0] if indata.ndim > 1 else indata
        self.audio_queue.put(audio_chunk.copy())

    def transcription_worker(self):
        """Worker thread for processing audio chunks"""
        print("Starting transcription worker...")

        # Create streaming context
        with self.model.transcribe_stream(
            context_size=(256, 256),  # Good balance for real-time
            depth=1,  # Fast processing
            keep_original_attention=False  # Use local attention for streaming
        ) as transcriber:

            while self.is_recording:
                try:
                    # Get audio chunk with timeout
                    audio_chunk = self.audio_queue.get(timeout=0.1)

                    # Convert numpy array to MLX array
                    audio_mx = mx.array(audio_chunk.astype(np.float32))

                    # Add audio to transcriber
                    transcriber.add_audio(audio_mx)

                    # Get current result
                    result = transcriber.result

                    # Update display if text changed
                    if result.text != self.current_text:
                        self.current_text = result.text
                        self.display_transcription()

                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"Transcription error: {e}")
                    continue

    def display_transcription(self):
        """Display current transcription with timestamp"""
        timestamp = datetime.now().strftime("%H:%M:%S")

        # Clear screen and show current transcription
        print("\033[2J\033[H", end="")  # Clear screen, move cursor to top
        print("=" * 80)
        print(f"🎤 LIVE LECTURE TRANSCRIPTION - {timestamp}")
        print("=" * 80)
        print()

        # Show current text
        if self.current_text.strip():
            print("Current transcription:")
            print("-" * 40)
            print(self.current_text)
            print()
        else:
            print("Listening... (speak into your microphone)")
            print()

        print("=" * 80)
        print("Press Ctrl+C to stop recording")
        print("=" * 80)

    def start_recording(self):
        """Start live audio recording and transcription"""
        print(f"Initializing audio capture...")
        print(f"Sample rate: {self.sample_rate} Hz")
        print(f"Chunk duration: {self.chunk_duration} seconds")
        print()

        self.is_recording = True

        # Start transcription worker thread
        self.transcription_thread = threading.Thread(target=self.transcription_worker)
        self.transcription_thread.daemon = True
        self.transcription_thread.start()

        try:
            # Start audio stream
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                callback=self.audio_callback,
                blocksize=self.chunk_size,
                dtype=np.float32
            ):
                print("🎤 Recording started! Speak into your microphone...")
                self.display_transcription()

                # Keep recording until interrupted
                while self.is_recording:
                    time.sleep(0.1)

        except KeyboardInterrupt:
            print("\n\n🛑 Recording stopped by user")
        except Exception as e:
            print(f"\n\n❌ Error: {e}")
        finally:
            self.stop_recording()

    def stop_recording(self):
        """Stop recording and save transcription"""
        self.is_recording = False

        if self.transcription_thread:
            self.transcription_thread.join(timeout=2.0)

        # Save final transcription
        if self.current_text.strip():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"lecture_transcription_{timestamp}.txt"

            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"Lecture Transcription - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 60 + "\n\n")
                f.write(self.current_text)

            print(f"\n💾 Transcription saved to: {filename}")

        print("\n✅ Session ended")


def main():
    """Main function to run live transcription"""
    print("🎓 Live Lecture Transcription")
    print("============================")
    print()

    # Check available audio devices
    print("Available audio devices:")
    print(sd.query_devices())
    print()

    try:
        # Initialize transcriber
        transcriber = LiveTranscriber(
            sample_rate=16000,  # Standard ASR sample rate
            chunk_duration=1.0  # 1 second chunks for responsiveness
        )

        # Start live transcription
        transcriber.start_recording()

    except Exception as e:
        print(f"❌ Failed to start transcription: {e}")
        print("Make sure you have a microphone connected and permissions enabled.")


if __name__ == "__main__":
    main()
