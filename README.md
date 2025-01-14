# LLM4EEG
This is the repository for the paper "Hidden States in LLMs improve EEG Representation Learning and Visual Decoding" at ECAI 2024. 

## Prepare for datasets.
The EEG data can be downloaded at [https://github.com/perceivelab/eeg_visual_classification](https://github.com/perceivelab/eeg_visual_classification).

The corresponding images can be downloaded at [https://image-net.org/](https://image-net.org/)

Put these files into the folder `Data/EEGDataset`

## Extract features from pretrained models.
Semantic features can be extracted from Llama-2 7b by inputting the description of the images `Data/EmbeddingFiles/image_des.txt`. 

Visual features can be extracted from VGG-19 by inputting the images into the model.

## Train the semantic encoder.
Train the semantic encoder by the clip loss 
`python training/EEG_sem_clip.py`
and finetune for classification `python training/EEG_sem_classification.py`. 
The corresponding weights are in the folder `Weights/EEG_Sem_Emb` and the folder `Weights/EEG_classification`. 

## Train the visual encoder.
Train the visual encoder by the clip loss `python training/EEG_vis_clip.py`.
The corresponding weights are in the folder `Weights/EEG_Vis_Emb`. 

## Generate images by glide.
Train an mlp model, which maps the semantic embeddings to the glide embedding space `python training/EEG2Glide.py`, and generate images `python training/generator.py`.

You can directly generate images with the weights used in our paper:
```
python training/pretrained_generator.py --subject 1
```

## Evaluate the results.
Calculate the metrics 
`python evaluation/ClipScore.py` 
`python evaluation/InceptionScore.py`
`python evaluation/SSIM.py`.

