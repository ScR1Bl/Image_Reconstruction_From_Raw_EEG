# Dataset and split

## Source

The project uses **THINGS-EEG2** by Gifford, Dwivedi, Roig, and Cichy. It is a
large visual EEG dataset with 10 participants, 82,160 trials per participant,
and 16,740 image conditions drawn from the THINGS object-image database.

- Dataset: https://osf.io/3jk45/
- Project page: https://things-initiative.org/
- Paper: https://doi.org/10.1016/j.neuroimage.2022.119754
- Original code: https://github.com/gifale95/eeg_encoding

Follow the dataset's current terms and the stimulus-image licenses. This project
does not redistribute EEG archives or images.

## Expected local layout

```text
data/things_eeg2_osf/
  image_set/
    training_images.zip
    test_images.zip
    image_metadata.npy
  preprocessed/
    sub-01.zip
    ...
    sub-10.zip
```

The preprocessed training tensor for every participant has shape
`[16540 images, 4 repetitions, 17 channels, 100 samples]`.

## Leakage-safe split

The official training partition provides ten distinct images for each of 1,654
concepts. We deterministically assign images 1-8 to train, image 9 to validation,
and image 10 to the untouched test. All four repetitions and all participant
responses for one image remain in the same split.

Create the index after downloading the dataset:

```bash
eeg-build-index
eeg-build-cache
```

The resulting split contains 661,600 EEG rows:

- train: 529,280 rows / 13,232 unique images;
- validation: 66,160 rows / 1,654 unique images;
- untouched test: 66,160 rows / 1,654 unique images.

The official 200-image THINGS-EEG2 test partition is separate from this internal
image holdout and should only be used for the dedicated 200-way benchmark.

## Citation

```bibtex
@article{gifford2022things_eeg2,
  title   = {A large and rich EEG dataset for modeling human visual object recognition},
  author  = {Gifford, Alessandro T. and Dwivedi, Kshitij and Roig, Gemma and Cichy, Radoslaw M.},
  journal = {NeuroImage},
  volume  = {264},
  pages   = {119754},
  year    = {2022},
  doi     = {10.1016/j.neuroimage.2022.119754}
}
```
