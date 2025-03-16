## 3. Evaluation
Traditional evaluation of VLMs typically relies on struc-
tured benchmark datasets where the model’s output is com-
pared against a predefined ground-truth answer. To com-
prehensively evaluate a VLM, we focus on five key bench-
marks—grammatical correctness, consistency, creativity,
meaningfulness, and plot—each of which plays a crucial
role in determining the model’s ability to generate struc-
tured and engaging descriptions. Our primary objective is to
investigate whether a VLM with as few as 6M–25M param-
eters can still generate coherent and contextually relevant
text. Inspired by the evaluation framework of (Eldan &
Li, 2023), we employ an LLM-based evaluation approach
that leverages GPT-4o to assess generated text quality. Our
evaluation setup consists of a manually curated dataset of
25 image descriptions, where each description’s beginning
along with its corresponding image, is provided as a prompt
to NanoVLMs. The model then completes the partial text
while attending to the image, and its output is subsequently
graded using Prompt 3 (shown in Figure 2) by GPT-4o based
on key evaluation benchmarks outlined in Table 3. To en-
sure that our image description generation task is non-trivial,
we deliberately structure the input prompts of 6–7 words
long for short descriptions and 18–20 words long for long
descriptions. This approach challenges the model’s abil-
ity to produce semantically meaningful and grammatically
sound completions, especially when required to infer miss-
ing context from the provided image. Furthermore, to verify
that the model does not simply memorize training data, we
conduct an analysis using ROUGE scores which is detailed
in the section 4. By integrating these methods, we pro-
vide a comprehensive assessment of NanoVLMs’ linguistic
and contextual competence, addressing the limitations of
traditional benchmark-driven approaches.