# varKoder train

After *varKodes* are generated with `varKoder image`, they can be used to train a neural network to recognize taxa based on these images. The `varKoder train` command uses `fastai` and `pytorch` to do this training, with image models obtained with the `timm` library. This includes a very large collection of models available on [Hugging Face Hub](https://huggingface.co/docs/hub/timm)

If a model supported by the `timm` library requires a specific input image size (for example, [vision transformers](https://huggingface.co/google/vit-base-patch16-224-in21k)), **varKoder** will automatically resize input **varkodes** using the [nearest pixel method](https://pillow.readthedocs.io/en/stable/handbook/concepts.html#PIL.Image.Resampling.NEAREST).

There are two modes of training:

 1. Multi-label
 
 This is the default training mode. With [multi-label classification](https://en.wikipedia.org/wiki/Multi-label_classification), each *varKode* can be associated with one or more labels. For example, you can use this to provide more information about the sample in addition to its identification, or to provide a full taxonomic hierarchy instead of a single name. In our tests, we have found that we can increase accuracy of predictions by flagging samples likely to have low DNA quality.
 
 2. Single label
 
  Our initial tests were all done with single-label classification. Even though we found limitation with this mode, we keep the option to do it for compatibility. In this case, it is not possible to account for sample quality when making predictions. To enable single label classification, you have to use options `--single-label` and `--ignore-quality`. 


## Arguments

### Required arguments

| argument | description |
| --- | --- |
| input       |          path to the folder with input images. varKoder will search for images in all subfolders. These images must have been generated with `varKoder image` |
| outdir       |         path to the folder where trained model will be stored. |


### Optional arguments

| argument | description |
| --- | --- |
| -h, --help | show help message and exit |
| -d SEED, --seed SEED | random seed passed to `pytorch`. |
| -x , --overwrite | overwrite existing results. |
| `-vv`, `--version` |  shows varKoder version. |
| -n NUM_WORKERS, --num-workers NUM_WORKERS | number of CPUs used for data loading. See https://docs.fast.ai/data.load.html#dataloader. The default (0) uses the main process. |
| -t LABEL_TABLE, --label-table LABEL_TABLE | path to csv table with labels for each sample. This table must have columns `sample` and `labels`. Labels are passed as a string, with multiple labels separated by `;`. By default, varKoder will attempt to read labels from image metadata instead of a label table, providing a label table overrides this behavior. Samples not present in label table will be ignored for training and validation. |
| -S, --single-label  |  Train as a single-label image classification model. This option must be combined with --ignore-quality. By default, models are trained as multi-label. |
| -d THRESHOLD, --threshold THRESHOLD | Confidence threshold to calculate validation set metrics during training. Ignored if using --single-label (default: 0.7) |
| -V VALIDATION_SET, --validation-set VALIDATION_SET | comma-separated list of sample IDs to be included in the validation set, or path to a text file with such a list. If not provided, a random validation set will be created. See `--validation-set-fraction` to choose the fraction of samples used as validation. |
| -f VALIDATION_SET_FRACTION, --validation-set-fraction VALIDATION_SET_FRACTION | fraction of samples to be held as a random validation set. If using multi-label, this applies to all samples. If using single-label, this applies to each species. (default: 0.2) |
| -m PRETRAINED_MODEL, --pretrained-model PRETRAINED_MODEL | pickle file with optional pretrained neural network model to update with new images. Overrides --architecture and --pretrained-timm |
| -b MAX_BATCH_SIZE, --max-batch-size MAX_BATCH_SIZE | maximum batch size when using GPU for training. (default: 64) |
| -r BASE_LEARNING_RATE, --base_learning_rate BASE_LEARNING_RATE | base learning rate used in training. See https://walkwithfastai.com/lr_finder for information on learning rates. (default: 0.005) |
| -e EPOCHS, --epochs EPOCHS | number of epochs to train. See https://docs.fast.ai/callback.schedule.html#learner.fine_tune (default: 30) |
| -z FREEZE_EPOCHS, --freeze-epochs FREEZE_EPOCHS | number of freeze epochs to train. Recommended if using a pretrained model, but probably unnecessary if training from scratch. See https://docs.fast.ai/callback. schedule.html#learner.fine_tune (default: 0) |
| -c ARCHITECTURE, --architecture ARCHITECTURE | model architecture. See below for details of possible options. (default: hf-hub:brunoasm/vit_large_patch32_224.NCBI_SRA)|
| -i NEGATIVE_DOWNWEIGHTING, --negative_downweighting NEGATIVE_DOWNWEIGHTING | Parameter controlling strength of loss downweighting for negative samples. See gamma(negative) parameter in https://arxiv.org/abs/2009.14119. Ignored if used with --single-label. (default: 4) |
| -w, --random-weigths | start training with random weigths. By default, pretrained model weights are downloaded from timm. See https://github.com/rwightman/pytorch-image-models. (default: False) |
| -M, --no-metrics | skip calculation of validation loss and metrics (default: False) |
| -X MIX_AUGMENTATION, --mix-augmentation MIX_AUGMENTATION | apply MixUp or CutMix augmentation. Possible values are `CutMix`, `MixUp` or `None`. See https://docs.fast.ai/callback.mixup.html (default: MixUp) |
| -s, --label-smoothing | turn on Label Smoothing. Only applies to single-label. See https://github.com/fastai/fastbook/blob/master/07_sizing_and_tta.ipynb (default: False) |
| -p P_LIGHTING, --p-lighting P_LIGHTING | probability of a lighting transform. Set to 0 for no lighting transforms. See https://docs.fast.ai/vision.augment.html#aug_transforms (default: 0.75) |
| -l MAX_LIGHTING, --max-lighting MAX_LIGHTING | maximum scale of lighting transform. See https://docs. fast.ai/vision.augment.html#aug_transforms (default: 0.25) |
| -g, --no-logging  | hide fastai progress bar and logging during training. These are shown by default. | 

## Model architecture

We support three possible inputs here:

1. Models supported by the timm library

You can use as input the name of a model supported by the timm library (see https://huggingface.co/docs/timm/quickstart for details). For example, this will use timm library to train a model using the resnet50 architecture pulled from timm library:

`varKoder train --architecture resnet50 input_dir output_dir`

2. Models from Hugging Face hub

The timm library allows downloading models from Hugging face hub directly. See
https://huggingface.co/docs/hub/timm for possible options. To pull a model, prepend 'hf-hub:' to the model repository on Hugging Face Hub. For example, the following will use the vit_large_patch32_224 architecture pretrained with varKodes produced from NCBI data:
`varKoder train --architecture hf-hub:brunoasm/vit_large_patch32_224.NCBI_SRA input_dir output_dir`

3. Models previously used with chaos game representations

We implemented two models previously applied to chaos game representations. In both cases, images are linearized instead of being treated as 2D images. 

To use the model employed by [DeLUCS](https://github.com/Kari-Genomics-Lab/iDeLUCS), use option `arias2022`:
`varKoder train --architecture idelucs input_dir output_dir`

To use the model employed by Fiannaca (2018), use option `fiannaca2018`:
`varKoder train --architecture fiannaca input_dir output_dir`

In both cases, there are no pretrained models available, you will have to train starting from random weights.

These are the references for both models:

1. Arias PM, Alipour F, Hill KA, Kari L (2022) DeLUCS: Deep learning for unsupervised clustering of DNA sequences. PLOS ONE, 17(1):e0261531. https://doi.org/10.1371/journal.pone.0261531

2. Fiannaca A, La Paglia L, La Rosa M, Lo Bosco G, Renda G, Rizzo R, Gaglio S, Urso A (2018) Deep learning models for bacteria taxonomic classification of metagenomic data. BMC Bioinformatics, 19(Suppl 7)https://doi.org/10.1186/s12859-018-2182-6

## GPU usage

If your system has more than one NVIDIA GPU, varKoder will by default use all available GPUs for training. If this behavior is undesirable, you need to set the visible GPUs before starting varKoder (see https://developer.nvidia.com/blog/cuda-pro-tip-control-gpu-visibility-cuda_visible_devices/). For example, the following command will call varKoder with a single visible GPU:

```bash
export CUDA_VISIBLE_DEVICES=0
varKoder train images_folder outfolder 
```

## Train command tips

All optional arguments are set to defaults that seemed to work well in our tests. Here we give some tips that may help you to modify these defaults

 1. The maximum batch size can be increased until the limit that your GPU memory supports. Larger batch sizes increase training speed, but might need adjustment of the base learning rate (which typically have to increase for large batch sizes as well). When there are only a few images available in the training set, `varKoder` automatically decreases batch size so that each training epoch sees about 10 batches of images. In our preliminary tests, we found that this improved training of these datasets.
 2. The number of epochs to train depends on the problem. Models trained for too long may overfit: be very good at the specific training set but bad a generalizing to new samples. Check the evolution of training and validation loss during training: if the training loss decreases but validation loss starts to increase, this means that your model is overfitting and you are training for too long. Because we introduce random noise during training with MixUp and lighting transformations, models should rarely overfit.
 3. Frozen epochs are epochs of training in which the deeper layers of a model are frozen (i. e. cannot be updated). Only the last layer is updated. Intuitively, it means that we keep the possible features that a model can recognize in an image fixed, and only update how it weigths these many different features to make a prediction. This can be useful if you have a model pretrained with `varKodes` and want to use transfer learning (i. e. update a trained model instead of start from scratch). We did not find transfer learning useful when models where previously trained with other kinds of images. However, we did find useful to pretrain some models as single label and use transfer learning to fine tune with frozen epochs as multilabel. 
 4. Finding a good learning rate is also somewhat of an art: if learning rates are too small, a model can get stuck in local optima or take too many epochs to train, wasting resources. If they are too large, the training cycle may never be able to hone into the best model weights. Our default learning rate (5e-3) behaves well for the varKodes that we used as test, but you may consider changing it in the following cases:
   1. If using a pretrained model, you may want to decrease the learning rate, since you expect to be closer to the optimal weigths already.
   2. If using a much larger batch size, you may want to increase the learning rate.
 5. For multi-label models, we use an asymmetric loss function that enables to weight differently positive and negative labels. This may have a major impact when a dataset is very large and there are many negative labels (for example, 99.9% of the varKodes do not have a label for a particular species). We found that a value o `4` worked in most cases, but this might be dataset-dependent and worth testing. This can be set with option `-g`.
 6. There is a wide array of possible model architectures, and new models come up all the time. You can use this resource to explore potential models: https://rwightman.github.io/pytorch-image-models/results/. The model we chose (vision transformer) was the most accurate among those that we tested that could be trained in a reasonable amount of time with the hardware we had in hand (M1 macs, NVIDIA A100 and A5000 GPUs). Generally, larger models will be more accurate but need more compute resources (GPU memory and processing time).
 7. In the paper, we found that a combination of CutMix with random lighting transforms (brightness and contrast) improves training and yields more accurate models for single-label models. MixUp had a similar performance to CutMix, and it seemed to work much better for multi-label classification. For this reason, MixUp and lighting transforms are turned on by default, but you can turn them off or even change the probability that a lighting transform is applied to a *varKode* during training. We also tested Label Smoothing, which was not as helpful. For this reason, it is turned off by default but can be turned on if desired.

During training, fastai outputs a log with some information (unless you use the `-g` option). This is a table showing, for each training epoch, the loss in images in the training set (`train_loss`), the loss in images in the validation set (`valid_loss`), the accuracy in images in the validation set. In the case of multi-label models, accuracy is measured as [area under ROC curve](https://en.wikipedia.org/wiki/Partial_Area_Under_the_ROC_Curve) and also [precision and recall](https://en.wikipedia.org/wiki/Precision_and_recall) using the provided confidence threshold for predictions and ignoring DNA quality labels. In the case of single-label models, we report `accuracy`, which is the fraction of varKodes for which the correct label is predicted. In each epoch, the model is presented and updated with all images in the training set, split in a number of batches according to the chosen batch size. The loss is a number calculated with a loss function, which basically shows how well a neural network can predict the labels of images it is presented with. It is expected that the train loss will be small, since these are the very same images that are used in training, and what you want is to see a small validation loss and large validation accuracy, since this shows how well your model can generalize to unseen data.

## Output

At the end of the training cycle, three files will be written to the output folder selected by the user:
 - `trained_model.pkl`: the model weights exported using `fastai`. This can be used as input again using the `--pretrained-model` option in case you want to further train the model or improve it with new images.
 - `labels.txt`: a text file with the labels that can be predicted using this model.
 - `input_data.csv`: a table with information about varKodes used in the training and validation sets.

## Examples

Here are several examples demonstrating how to use the `train` command for different training scenarios:

### Example 1: Basic Multi-label Training

Train a model using the default architecture (vision transformer) with multi-label classification:

```bash
varKoder train path/to/images trained_model
```

This uses the default vision transformer architecture (hf-hub:brunoasm/vit_large_patch32_224.NCBI_SRA) to train a multi-label classification model on the varKodes in the images folder, with a 0.7 confidence threshold for validation metrics.

### Example 2: Training with a Different Architecture

Train a model using a ResNet50 architecture:

```bash
varKoder train path/to/images resnet50_model --architecture resnet50
```

This trains a model using the ResNet50 architecture from the timm library instead of the default vision transformer, which may train faster but might have lower accuracy.

### Example 3: Single-label Classification with Label Smoothing

Train a model for single-label classification with label smoothing:

```bash
varKoder train path/to/images single_label_model --single-label --ignore-quality --label-smoothing
```

This trains a model for single-label classification (each sample assigned exactly one label) with label smoothing to help prevent overfitting. The `--ignore-quality` flag is required with single-label mode.

### Example 4: Freezing Layers and Transfer Learning

Fine-tune a pretrained model by first training only the final layer:

```bash
varKoder train path/to/images transfer_model --pretrained-model path/to/existing_model.pkl --freeze-epochs 5 --epochs 20
```

This loads an existing model and trains it with 5 epochs where only the last layer is updated (frozen epochs), followed by 15 epochs where all layers are updated, which is useful for transfer learning.

### Example 5: Customizing Training Parameters

Train with custom batch size, learning rate, and data augmentation settings:

```bash
varKoder train path/to/images custom_model --max-batch-size 32 --base_learning_rate 0.001 --mix-augmentation CutMix --p-lighting 0.5 --max-lighting 0.2
```

This trains a model with a smaller batch size and learning rate, uses CutMix instead of MixUp for data augmentation, and applies less aggressive lighting transformations during training.
