# 2.1. Dataset
## 2.1.1. DATASET OVERVIEW
To train NanoVLMs, we utilized the COCO (Common
Objects in Context)(Lin et al., 2014; Chen et al., 2015)
dataset, a widely recognized resource in computer vision
tasks. This dataset is ideal for our study as it features high-
resolution, richly annotated images from diverse domains
including people, animals, food, vehicles, and outdoor set-
tings—perfectly aligning with the learning analogy of 3–4
year old children. For our work, we specifically leveraged
the image-captioning component of COCO, where each im-
age is paired with five natural language captions describing
the scene and its objects. From this dataset, we selected
approximately 28K image-caption pairs, using 90 percent
for training and 10 percent for validation. Additionally, to
evaluate the model’s knowledge, versatility, and generaliza-
tion capabilities, we tested it on 25 separate data samples
that were entirely distinct from the training and validation
sets.
## 2.1.2. DATASET PREPARATION
To prepare the dataset, we used the COCO dataset’s images
and corresponding captions to generate image descriptions,
constructing two datasets: ShortDesc and LongDesc. Specif-
ically, ShortDesc comprises concise image descriptions of
20–25 words, while LongDesc features detailed image de-
scriptions of 60–70 words. These datasets were designed
to assess how the model handles shorter versus longer text
inputs, reflecting its ability to process and generate meaning-
ful and consistent outputs. This mirrors the developmental
process of 3–4 year old children, who acquire intellectual
abilities through exposure to diverse visuals along with vari-
ous linguistic patterns.

For generating these descriptions, we employed OpenAI’s
GPT-4o, a SOTA text generation model capable of produc-
ing high-quality synthetic content. Combined captions for
each image (all captions) along with the respective prompt is
passed to GPT-4o(OpenAI, 2024), where Prompt1 shown in
Figure 2 is used to generate ShortDesc dataset and Prompt2
shown in Figure 2 is used to generate LongDesc dataset.
The model produced outputs based on the structure and
constraints defined in the respective prompts. Figure 3 illus-
trates the process of prompt passing and dataset preparation.