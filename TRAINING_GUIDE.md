# Training Strategy & Learnings for LLM-Backbone TTS

---

## 1. The Prompt Format & Token Specialization
Early experiments showed that simply feeding text and discrete audio tokens without clear boundaries leads to confusion. The LLM struggles to differentiate between "understanding text" and "generating audio representations."

**Learning:** We heavily customized the prompt and tokenizer to include explicit, distinct special tokens. 
- The format must explicitly anchor the model's intent: `[TEXT]<text prompt>[START]<audio tokens>[STOP]`.
- We added dedicated special tokens (`<|audio_start|>`, `<|audio_end|>`, `<|text|>`, etc.) to the tokenizer vocabulary and resized the embedding matrices. This provides absolute clarity to the causal transformer regarding which modality it is currently processing.

---

## 2. LLM Training Stability Strategies
Teaching a causal language model (Qwen backbone) to map text directly to audio-latent distributions often results in the model outputting garbage audio if not carefully stabilized.

**Learning & Strategy:** The loss calculation must be incredibly intentional.
*   **Batching:** Changed the batching from packing to dynamic batching - one sample per row. Felt like this could be more stable. Shall experiment training with packing. I think this should also work. 
*   **Text Weighting:** Initially trained the llm with text_weight as 0, but turns out that the model wasn't able to learn this way. It was overfitting and wasn't able to generalize. I was guessing even with 0 weights for text tokens, the model should be able to learn the relationship between text and audio tokens and generalize well.
If the text tokens are zero-weighted in the loss calculation aggressively from the start, the model can lose its language understanding capabilities, learning only to spout random audio formats. A critical learning is to experiment with text weighting strategies (e.g., maintaining a small loss weight on the text prediction part for the first N steps) so the LLM retains language comprehension and can generalize on new texts.
*   **Masking for Loss:** Ensure that padding tokens (`[PAD]`) or masked sections are strictly ignored in the `CrossEntropyLoss` calculation. Any gradient penalty on padding will rapidly destabilize autoregressive predictions.
*   **Gradual Length Training (Curriculum Learning):** The model often crashes or learns poorly if immediately exposed to full 1000+ token sequences. It's recommended starting with shorter text-audio pairs and gradually increasing the sequence context length (`max_length`) as the model stabilizes. Yet to try this. 

---

## 3. Two-Stage Decoder (Vocoder) Training
The role of the Decoder (Vocos) is to take the *continuous* hidden states evaluated by the LLM and construct high-fidelity audio waveforms.

**Learning & Strategy:** The Decoder must be trained independently in two distinct stages, with the LLM rigidly completely frozen.

#### Stage 1: Pure Reconstruction (Global Structure)
*   **Objective:** Teach the Decoder how to map hidden state dimensions to spectral layout.
*   **Method:** Train using purely reconstruction losses: **Mel-Spectrogram L1 Loss** and **Multi-Resolution STFT Loss**.
*   **Data:** This phase is executed over the *almost-full* generated sequences to ensure the Decoder learns long-term structural coherence and timing alignment.

#### Stage 2: Adversarial Refinement (Local Fidelity)
*   **Objective:** Eliminate robotic artifacts and "muffled" qualities to achieve crystal-clear, natural acoustic texture.
*   **Method:** Introduce Discriminator networks (Multi-Period and Multi-Scale).
*   **Crucial Learning (Cropping):** Running Discriminators (GANs) on full-length sequences destroys memory and stability. We transitioned to **Segment-Based Training (Random Cropping)**. During this stage, we extract random 1-second chunks (e.g., ~32,000 samples) from both the ground truth and the generated audio, feeding only these small crops into the Discriminator.
*   **Combined Loss:** The final phase loss comprises Reconstruction (Full Audio) + Adversarial GAN Loss (Cropped Audio) + Discriminator Feature Matching.