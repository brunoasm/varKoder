# varKoder.py train

After *varKodes* are generated with `varKoder.py image`, they can be used to train a neural network to recognize taxa based on these images. The `varKoder.py train` command uses `fastai` and `pytorch` to do this training, with image models obtained with the `timm` library. If a model in the `timm` library requires a specific input image size (for example, [vision transformers](https://huggingface.co/google/vit-base-patch16-224-in21k)), **varKoder** will automatically resize input **varkodes** using the [nearest pixel method](https://pillow.readthedocs.io/en/stable/handbook/concepts.html#PIL.Image.Resampling.NEAREST).

There are two modes of training:

 1. Multi-label
 
 This is the default training mode. With [multi-label classification](https://en.wikipedia.org/wiki/Multi-label_classification), each *varKode* can be associated with one or more labels. For example, you can use this to provide more information about the sample in addition to its identification, or to provide a full taxonomic hierarchy instead of a single name. In our tests, we have found that we can increase accuracy of predictions by flagging samples likely to have low DNA quality.
 
 2. Single label
 
  Our initial tests were all done with single-label classification. Even though we found limitation with this mode, we keep the option to do it for compatibility. In this case, it is not possible to account for sample quality when making predictions. To enable single label classification, you have to use options `--single-label` and `--ignore-quality`. 


## Arguments

### Required arguments

| argument | description |
| --- | --- |
| input       |          path to the folder with input images. These must have been generated with `varKoder.py image` |
| outdir       |         path to the folder where trained model will be stored. |


### Optional arguments

| argument | description |
| --- | --- |
| -h, --help | show help message and exit |
| -d SEED, --seed SEED | random seed passed to `pytorch`. |
| -t LABEL_TABLE, --label-table LABEL_TABLE | path to csv table with labels for each sample. If not provided, varKoder will attemp to read labels from the image file metadata. |
| -i, --ignore-quality |  ignore sequence quality when training. By default low-quality samples are labelled as such. |
|  -n, --single-label  |  Train as a single-label image classification model. This option must be combined with --ignore-quality. By default, models are trained as multi-label. |
|  -d THRESHOLD, --threshold THRESHOLD | Confidence threshold to calculate validation set metrics during training. Ignored if using --single-label (default: 0.7) |
| -V VALIDATION_SET, --validation-set VALIDATION_SET | comma-separated list of sample IDs to be included in the validation set. If not provided, a random validation set will be created. See `--validation-set-fraction` to choose the fraction of samples used as validation. |
| -f VALIDATION_SET_FRACTION, --validation-set-fraction VALIDATION_SET_FRACTION | fraction of samples to be held as a random validation set. If using multi-label, this applies to all samples. If using single-label, this applies to each species. (default: 0.2) |
| -m PRETRAINED_MODEL, --pretrained-model PRETRAINED_MODEL | pickle file with optional pretrained neural network model to update with new images. By default models are initialized with random weights. This option can be useful to update models as more samples are obtained. |
| -b MAX_BATCH_SIZE, --max-batch-size MAX_BATCH_SIZE | maximum batch size when using GPU for training. (default: 64) |
| -r BASE_LEARNING_RATE, --base_learning_rate BASE_LEARNING_RATE | base learning rate used in training. See https://walkwithfastai.com/lr_finder for information on learning rates. (default: 0.005) |
| -e EPOCHS, --epochs EPOCHS | number of epochs to train. See https://docs.fast.ai/ca llback.schedule.html#learner.fine_tune (default: 20) |
| -z FREEZE_EPOCHS, --freeze-epochs FREEZE_EPOCHS | number of freeze epochs to train. Recommended if using a pretrained model, but probably unnecessary if training from scratch. See https://docs.fast.ai/callback. schedule.html#learner.fine_tune (default: 0) |
| -c ARCHITECTURE, --architecture ARCHITECTURE | model architecture. See https://github.com/rwightman/pytorch-image-models for possible options. (default: ig_resnext101_32x8d) |
| -P, --pretrained | download pretrained model weights from timm. See https://github.com/rwightman/pytorch-image-models. (default: False) |
| -X MIX_AUGMENTATION, --mix-augmentation MIX_AUGMENTATION | apply MixUp or CutMix augmentation. Possible values are `CurMix`, `MixUp` or `None`. See https://docs.fast.ai/callback.mixup.html (default: MixUp) |
| -s, --label-smoothing | turn on Label Smoothing. Only applies to single-label. See https://github.com/fastai /fastbook/blob/master/07_sizing_and_tta.ipynb (default: False) |
| -p P_LIGHTING, --p-lighting P_LIGHTING | probability of a lighting transform. Set to 0 for no lighting transforms. See https://docs.fast.ai/vision.a ugment.html#aug_transforms (default: 0.75) |
| -l MAX_LIGHTING, --max-lighting MAX_LIGHTING | maximum scale of lighting transform. See https://docs. fast.ai/vision.augment.html#aug_transforms (default: 0.25) |
| -g, --no-logging  | hide fastai progress bar and logging during training. These are shown by default. | 

## Train command tips

All optional arguments are set to defaults that seemed to work well in our tests. Here we give some tips that may help you to modify these defaults

 1. The maximum batch size can be increased until the limit that your GPU memory supports. Larger batch sizes increase training speed, but might need adjustment of the base learning rate (which typically have to increase for large batch sizes as well). When there are only a few images available in the training set, `varKoder` automatically decreases batch size so that each training epoch sees about 10 batches of images. In our preliminary tests, we found that this improved training of these datasets.
 2. The number of epochs to train is somewhat of an art. Models trained for too long may overfit: be very good at the specific training set but bad a generalizing to new samples. Check the evolution of training and validation loss during training: if the training loss decreases but validation loss starts to increase, this means that your model is overfitting and you are training for too long. Because we introduce random noise during training with MixUp and lighting transformations, models should rarely overfit.
 3. Frozen epochs are epochs of training in which the deeper layers of a model are frozen (i. e. cannot be updated). Only the last layer is updated. This can be useful if you have a model pretrained with `varKodes` and want to use transfer learning (i. e. update a trained model instead of start from scratch). We did not find transfer learning useful when models where previously trained with other kinds of images.
 4. Finding a good learning rate is also somewhat of an art: if learning rates are too small, a model can get stuck in local optima or take too many epochs to train, wasting resources. If they are too large, the training cycle may never be able to hone into the best model weights. Our default learning rate (5e-3) behaves well for the varKodes that we used as test, but you may consider changing it in the following cases:
   1. If using a pretrained model, you may want to decrease the learning rate, since you expect to be closer to the optimal model already.
   2. If using a much larger batch size, you may want to increase the learning rate.
 5. There is a wide array of possible model architectures, and new models come up all the time. You can use this resource to explore potential models: https://rwightman.github.io/pytorch-image-models/results/. The model we chose (ig_resnext101_32x8d) was the most accurate among those that we tested. Generally, larger models will be more accurate but need more compute resources (GPU memory and processing time).
 6. In the paper, we found that a combination of CutMix with random lighting transforms (brightness and contrast) improves training and yields more accurate models for single-label models. MixUp had a similar performance to CutMix, and it seemed to work much better for multi-label classification. For this reason, MixUp and lighting transforms are turned on by default, but you can turn them off or even change the probability that a lighting transform is applied to a *varKode* during training. We also tested Label Smoothing, which was not as helpful. For this reason, it is turned off by default but can be turned on if desired.

During training, fastai outputs a log with some information (unless you use the `-g` option). This is a table showing, for each training epoch, the loss in images in the training set (`train_loss`), the loss in images in the validation set (`valid_loss`), the accuracy in images in the validation set. In the case of multi-label models, accuracy is measured as [precision and recall](https://en.wikipedia.org/wiki/Precision_and_recall) using the provided confidence threshold for predictions and ignoring DNA quality labels. In the case of single-label models, we report `accuracy`, which is the fraction of varKodes for which the correct label is predicted. In each epoch, the model is presented and updated with all images in the training set, split in a number of batches according to the chosen batch size. The loss is a number calculated with a loss function, which basically shows how well a neural network can predict the labels of images it is presented with. It is expected that the train loss will be small, since these are the very same images that are used in training, and what you want is to see a small validation loss and large validation accuracy, since this shows how well your model can generalize to unseen data.

## Output

At the end of the training cycle, three files will be written to the utput folder selected by the user:
 - `trained_model.pkl`: the model weigths exported using `fastai`. This can be used as input again using the `--pretrained-model` option in cae you want to further train the model or improve it with new images.
 - `labels.txt`: a text file with the labels that can be predicted using this model.
 - `input_data.csv`: a table with information about varKodes used in the training and validation sets.