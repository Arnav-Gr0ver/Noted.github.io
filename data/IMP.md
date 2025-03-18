# Data processor

a module (make sure this key word is right) that handles loading dataset split

# preprocess

preprocess train / val splits & annotations into percentages given in paper

paper does not specifically state which year dataset they use, so we use the latest images and annotations from 2017

paper states they use image captioning component of dataset so we use train/val/test split with standard annotations (no stuff panoptic etc.)

paper uses 28k image caption pairs + 25 (90% for training so 25.2k, and 10% for validation so 2.8k + 25 selected from testing)

paper does not go into how they went about getting the original 28k neither the selection process for obtaining the splits; we will use our best assumptions and use (XYZ method) for selecting image-caption pairs - with each percentage split coming from their independent given ds splits. (i.e. 25.2 train samples from larger train 118k - basically filtering it down)