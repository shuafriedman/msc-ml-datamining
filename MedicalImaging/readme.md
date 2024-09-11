EVA: Basic VIT architecture (small nuance differences). Pre-training on unlabled image data using MIM.
The image is run through 2 parallel phases: 1) Passing the full image into the CLIP L/14 vision tower, which splits the image into 14X14 patches, and creates a continuous embedding for each patch. The CLIP model is pre-trained on image-text pairs, so these embeddings contain rich pre-trained context from text as well (emphasis, the pretraining for EVA itself doesn't utilize text. The text reffered to here is from the CLIP pre-training). 2) The full image is split into patches, and masked 40%. This is fed into the EVA model, with the goal of trying to predict the masked embeddings for the corresponding CLIP patches.

This is in constrast to a model like BeiT, which uses a discrete learned tokenizer, like BERT, for the patches. Meaning, instead of tokens, the model here is trying to predict a continuous vector.


https://huggingface.co/timm/eva_giant_patch14_560.m30m_ft_in22k_in1k

https://arxiv.org/pdf/2106.08254 (BeiT, good image in the beginning)

