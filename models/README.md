## 2.2. Architecture
The primary objective of NanoVLMs is to complete par-
tially provided textual descriptions by generating coherent
and contextually appropriate outputs. To achieve this, we de-
signed a VLM with a simple yet effective transformer-based
architecture consisting of three key components: a visual
encoder for processing images, a visual-textual connector to
bridge visual and textual modalities, and a language decoder
for generating text as shown in Figure 1.
The core of the NanoVLM architecture lies in its transformer
blocks (shown in Figure 5), which form the foundation of
both the visual encoder and the language decoder. Each
transformer block comprises multi-head attention(Jalammar,
2019; Vaswani et al., 2023) for capturing relationships
across input tokens—whether image patches or text—and
a multi-layer perceptron (MLP) for processing the outputs
of the attention mechanism. To ensure stable training and
faster convergence, layer normalization is applied prior to
the attention and MLP layers. A key distinction in the de-
coder is the use of causal self-attention, where masking is
employed to uphold the autoregressive nature of text gener-
ation. This mechanism is vital for maintaining coherence
and contextual accuracy, ensuring that predictions are based
solely on prior information, a critical requirement for gener-
ating fluent and logically consistent textual descriptions.
## 2.2.1. VISUAL ENCODER
The visual encoder in NanoVLM is a critical component
responsible for extracting meaningful features from im-
ages, drawing inspiration from the Vision Transformer (ViT)
architecture while being optimized for compactness. To
maintain performance, we process images at a resolution
of 224x224 pixels(Thapa et al., 2024), dividing them into
16x16 pixel patches to yield 196(Wen et al., 2024) patches
per image. These patches undergo a series of transforma-
tions beginning with patch embedding, where the image is
passed through two 2D convolutional layers(as shown in
Figure 4) followed by layer normalization(Ba et al., 2016)
and ReLU(Agarap, 2019) activation. This is succeeded
by a fully connected neural network, which transforms the
patches into 196 tokens. A [CLS] token is then prepended,
making the sequence 197 tokens. Positional encoding is
applied to retain spatial information, followed by normaliza-
tion. These enriched embeddings are then processed through
a series of transformer blocks, where multi-head attention
mechanisms capture contextual dependencies between the
patches. Finally, the [CLS] token is aggregated to form a
compact representation that encapsulates the salient features
of the image. This streamlined yet robust approach ensures
effective visual feature extraction while keeping the model
size minimal.
## 2.2.2. VISUAL-TEXTUAL CONNECTOR
The visual-textual connector is a pivotal component in the
NanoVLMs architecture, responsible for bridging the gap
between the visual and textual modalities. The visual em-
beddings and the textual embeddings must be aligned in
the same dimensional space to enable effective interaction
between the two modalities. To achieve this, we employ
a multimodal projector that consists of a single learnable
layer followed by GELU that reduces the dimensionality of
the visual embeddings. Once the visual embeddings are pro-
jected into the textual embedding space, both the visual and
textual embeddings are concatenated to form a multimodal
token embedding. This combined representation effectively
encapsulates both the image’s content and its corresponding
textual description. The resulting multimodal token embed-
ding is then passed as input to the decoder block, where
it will guide the generation of coherent and contextually
relevant textual descriptions.
## 2.2.3. DECODER BLOCK
The decoder block in NanoVLM transforms fused visual-
textual embeddings into coherent text using a transformer-
based architecture, ensuring text generation. It begins by
passing the multimodal token embedding through a posi-
tional embedding layer, which encodes token order. The
input then moves through transformer blocks with multi-
head self-attention, but unlike the encoder, the decoder ap-
plies causal self-attention, masking(Liu et al., 2022; Yin
et al., 2024) future tokens to prevent information leakage
and enforce autoregressive generation. Finally, the pro-
cessed output undergoes layer normalization and a linear
projection, mapping it to a vocabulary space where logits
determine the next token. This structured decoding mecha-
nism enables NanoVLM to generate fluent, context-aware
descriptions when provided with both an image and par-
tial text as input.We employ cross-entropy loss to compute
the error between the predicted and actual target text. This
loss is used to guide the training of the model, optimizing
the parameters to generate accurate and coherent textual
descriptions.