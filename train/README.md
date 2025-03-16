## 2.3. Experiments
This section details the experimental setup and hyperparam-
eter tuning for training all three versions of NanoVLM. The
models were trained on a single A100 GPU, with key hyper-
parameters such as n blks (number of transformer blocks in
the visual encoder), n layer (number of transformer layers
in decoder), n head (number of attention heads), head size
(size of each head), n embd (textual embedding dimension),
and img embd dim (visual embedding dimension) gradu-
ally scaled up as we moved from the mini to the large ver-
sion, as shown in Table 1. This progressive scaling allowed
us to analyze how increasing model capacity influenced
performance while maintaining computational efficiency.
Certain hyperparameters remained fixed across all versions
to ensure stability during training, including dropout = 0.1,
image size = 224x224, patch size = 16x16, and learning
rate = 1e-3. Additionally, we present the distribution of
total learnable parameters for each version of the model
across the three core modules in Table 2. Since 3-4 year
old children primarily learn through visual cues, we allo-
cated a larger portion of the model’s parameters to the visual
encoder module, ensuring that the extracted features from
images were rich and informative while maintaining an effi-
cient balance between vision and language processing.