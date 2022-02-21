# MBRCE-AAE

<p align="center">
  <img src="https://user-images.githubusercontent.com/32597777/154869877-655ec5ce-3099-43ef-a583-1cbf5997766b.jpg" width="900"><br>
Example style transfer results for a matrix-based Rényi's cross-entropy adversarial autoencoder (MBRCE-AAE).  The goal is to transfer coarse facial features from the source images on the top row to the target images in the left column.  Here, we have trained the network on the CelebA face dataset at 256x256 resolution.
</p>

## Paper Overview
**Estimating Rényi's <img src="https://render.githubusercontent.com/render/math?math=\alpha">-Cross-Entropies in a Matrix-Based Way**<br>
Isaac J. Sledge and José C. Príncipe<br>
IEEE Transactions on Information Theory (under review) [arXiv: <a href="https://arxiv.org/abs/2109.11737">2109.11737</a>]<br>

**Abstract:** *Conventional information-theoretic quantities assume access to probability distributions.  Estimating such distributions is not trivial.  Here, we consider function-based formulations of cross entropy that sidesteps a priori estimation.  We propose three measures of Rényi's <img src="https://render.githubusercontent.com/render/math?math=\alpha">-cross-entropies in the setting of reproducing-kernel Hilbert spaces.  Each measure has its appeals.  We prove that we can estimate these measures in an unbiased, non-parametric, and minimax-optimal way using samples drawn from the unknown distributions.  We do this via sample-constructed Gram matrices.  This yields a matrix-based estimator of Rényi's <img src="https://render.githubusercontent.com/render/math?math=\alpha">-cross-entropies.  This estimator satisfies all of the axioms that Rényi established for divergences.  Our matrix-based cross-entropies can thus be used for assessing differences of arbitrary distributions.  They are also appropriate for handling high-dimensional distributions, since the convergence rate of our estimator is independent of the sample dimensionality.*<br>

**Findings:** A major advantage of our cross-entropy measures is that the convergence rate for the operator approximation depends only on the number of samples, not the sample dimensionality.  This property allows practitioners to consider assessing distributional overlap for high-dimensional samples where existing plug-in estimators, like Parzen windows, would be largely ineffective.  For the bipartite measures, this convergence-rate guarantee is available for all universal kernels and arbitrary distributions.  For the tripartite measure, we only have a dimensionally-agnostic error rate for radial universal kernels applied to samples from arbitrary distributions.  In either case, we have minimax optimality.  We thus avoid slow-convergence-rate issues that are prevalent for plug-in density estimators.<br>

Perhaps the biggest advantage of our measures is that they are suitable for any distribution.  They are not limited to just analyzing Gaussians, for example.  They are also not limited to analyzing unimodal distributions.<br>

We have shown that these properties are present in practice.  For instance, the measures empirically converge at essentially the same rate, regardless of the distribution dimensionality.  They also return cross-entropy magnitudes that are intuitively aligned with those from classical measures.<br>

In an extended set of experiments, we highlighted that our measures can be applied to deep learning.  We used them to generalize variational autoencoders to handle arbitrary priors.  This enabled us to consider multi-modal, adversarial priors within convolutional networks to model complicated imagery.<br>  

As a part of these experiments, we had several interesting findings.  We proved that our chosen priors are theoretically guaranteed to permit our autoencoders to behave like principled generative models.  That is, they can produce synthetic samples that are ensured to be indistinguishable from real samples.  Our empirical results lend credence to this claim.  They handily outperform variational autoencoders.  Likewise, our results are shown to be on par with those from recent generative-adversarial networks.  This is surprising in its own right.  Generative-adversarial networks are widely viewed to be far more capable than variational autoencoders at generating realistic synthetic data.  It is even more surprising that our network can do this without directly learning a sample discriminator.  Rather, it simply leverages the prior to simultaneously push away and pull back synthetically generated samples in a way that enhances their realism over time.  Without our information-theoretic measures, achieving such behavior would have been difficult and much less straightforward.<br>

Our extended experiments also reinforce manly of the claims that we made throughout.  They indicate that our measures can be used in very high-dimensional spaces where estimators like Parzen windows would be ineffective.  Our measures also work well with few samples.  They, additionally, can converge either as quickly or more quickly than dimensionally sensitive estimators.  We theoretically prove this claim, at least for the case where evidence-lower-bound optimization is used in conjunction with variational inference.<br>



## Prerequisites
This is as PyTorch implementation of our matrix-based Rényi's cross-entropy (MBRCE) estimator for use in creating arbitrary-prior adversarial autoencoders (MBRCE-AAEs). 

For convenience in training and applying MBRCE-AAEs, we have provided an Anaconda environment file, `mbrceaae-environ.yaml`.  Running the following command, 

    conda env create -f mbrceaae-environ.yaml

installs the required packages in a `conda` environment with the name `mbrceaee`.  This environment file assumes that the absolute path to your Anaconda installation is `~/anaconda3/`.  If you have installed Anaconda in a non-standard location, then you will need to modify the last line of `mbrceaae-environ.yaml` so that the `prefix` variable contains the proper path to it.

For those not wanting to use Anaconda, the file `mbrceaee-requiremnts.txt` provides a comprehensive list of needed packages. Run 

    pip3 install -r mbrceaee-requiremnts.txt

to install the latest versions of these packages and their dependencies.  For the major packages used by our codebase, we recommend the following versions: `torch >= 1.2`, `torchvision >= 0.4`, `numpy >= 1.17`, `scipy >= 1.3.1`, `opencv >= 3.4.2`, `sklearn >= 0.24`, and `imageio >= 2.14.1`.  For the minor packages, we recommend the following versions: `dareblopy >= 0.0.3`, `bimpy >= 0.1.1`, `pillow >= 7.2.0`, `dlutils >= 0.0.12`, `packaging >= 21.3`, `matplotlib >= 3.2.2`, `tqdm >= 4.47.0`, and `yacs >= 0.1.8`.  It may be possible to use slightly older versions than what are listed, but we have not tested this.  Likewise, the depencies for the major and minor packages need to be installed.  We recommend using `python >= 3.6`.


## MBRCE-AAE Dataset Preparation

Training the MBRCE-AAEs is done using TFRecords.  TFRecords are serialized containers of data that can be read linearly, which is useful for multi-node, multi-GPU training on a local network.

While TFRecords were, traditionally, exclusive to TensorFlow, the <a href="https://github.com/podgorskiy/DareBlopy">DareBlopy package</a> permits PyTorch to leverage them.

We have provided a script for converting images, in, say JPEG or PNG format, into TFRecords, `prepare_tfrecords.py`.  This script relies on a corresponding configuration file located within the `configs` directory in the repository root.  We have provided several configuration files.  If you wanted to prepare TFRecords for the CelebA Faces dataset, then the following command should be used from the repository root:

    python3 prepare_tfrecords.py -c ./configs/celeba-256x256.yaml

This script will output a series of TFRecords within the `datasets` folder at `/datasets/celeba-256x256-tfrecords/`.  Alternatively, if you modify the configuration file, then they will be located in whatever directory is specified.

Within `prepare_tfrecords.py`, we make several assumptions.  We assume that the images for a given dataset exist in a folder `datasets` located in the repository root.  For instance, in the case of the 30,000-image CelebA dataset, we assume a series of JPEG images are located at `datasets/celeba-256x256-images/`.  Each image within the `celeba-256x256-images` folder is numbered from `00001.jpg` to `30000.jpg` where zero-padded file naming is employed.  For arbitrary datasets, this naming scheme will likely not be used, so you can enforce it via the following shell commands from within a directory containing JPEG imagery:

    n=1; for file in *.jpg ; do mv "${file}" "${n}".jpg; ((n++)); done
    for f in *.jpg; do if [[ $f =~ [0-9]+\. ]]; then mv $f `printf "%.4d" "${f%.*}"`.jpg; fi; done

Each image is assumed to be square and has the same dimensions as all other images.  It is highly recommended that dimensions which are powers of two be used, which is due to our multi-resolution training procedure.  If your imagery is non-square, then it will need to be cropped.  ImageMagick is a good tool for batch cropping, e.g., `mogrify -gravity Center -crop 256x256+0+0 +repage "*.jpg"`.  We handle non-RGB color spaces, such as indexed colormaps, within `prepare_tfrecords.py`.  However, no guarantees are made that it will address all cases, so it is a best practice to save the imagery in an RGB-based format.  Again, ImageMagick is a good tool for batch pre-processing to achieve compliance.

Once the TFRecords are formed for a given dataset, there is no need to recreate them unless you either modify the original imagery or wish to consider additional image resolutions for network training.

Currently, `prepare_tfrecords.py` does not assume that the training imagery and test imagery are located in separate folders.  Rather, it assumes that the user specifies the number of training and testing samples to be drawn from a single directory specified within the configuration file.


## MBRCE-AAE Training

Once the TFRecords have been prepared, the MBRCE-AAEs can be trained using `train_mbrceaee.py`.  This script takes a dataset configuration file as a command-line input.  If you wanted to train an MBRCE-AEE on the CelebA Faces dataset, for instance, then the following command should be used from the repository root:

    python3 train_mbrceaee.py -c ./configs/celeba-256x256.yaml

Training will be performed on all local GPUs, up to a total of eight GPUs.  We assume, in the configuration file, that each compute node will be using up to a power-of-two number of GPUs and have thus specified training batch sizes for one GPU, two GPUs, four GPUs, and 8 GPUs.  If you only wish to leverage certain devices, then it would be good to specify them directly within `train_mbrceaee.py`.  For example, if you wanted to use the first four GPUs, then insert the line `import os; os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"` after the initial package import code block.

Some modification of `train_mbrceaee.py` is needed to handle network-based multi-node training.  However, we used PyTorch's distributed data parallel class for handling this case, so this is straightforward to do.  We have scaled to several hundred GPUs without issue.

During training, several data products will be exported to the `results` directory located in the repository root.  These include model checkpoint files, training logs, plots of the network performance, and examples of reconstructed and generated images.  

If the training process is interrupted for whatever reason, then `train_mbrceaee.py` can be run again with the appropriate configuration file.  It will continue from the last available model checkpoint, as specified in `last_checkpoint.txt`.  It is recommended that you not change the total number of GPUs in the middle of a run.

We train the MBRCE-AAEs in a hierarchical fashion based on increasing resolution levels.  Training starts at a very small image resolution, so the initial reconstructions and generated samples in the `results` directory will be quite small and thus nonsensical.  After a certain number of epochs have elapsed, which is specified by the configuration file, the image resolution will double.  This resolution-doubling process continues until the highest resolution is reached.  At this point, the MBRCE-AAEs should begin to model the imagery well and begin to synthesize realistic samples.  `train_mbrceaee.py` will therefore begin to call `metrics/fid_score.py` to compute the Fréchet inception distance (FID) every two epochs.  If the FID score has decreased, then a new model checkpoint will be created and stored in the `results` directory.

This code is designed to run on either CPUs or GPUs, but we overwhelmingly recommend the latter.  All of our models were simultaneously trained on multiple network nodes.  Each node ran `CUDA 11.0` and contained either eight NVIDIA Quadro RTX 8000 GPUs, with 48 GB VRAM, or eight NVIDIA Quadro RTX 6000 GPUs, with 24 GB VRAM.  Each node was connected via InfiniBand HDR with a bandwidth of 200 Gbps.  In total, we simultaneously used over a hundred GPUs for training.  Training a single model took a few hours.  

For training, you should use compute nodes with at least four recent GPUs on par with either an NVIDIA Quadro RTX 5000 or an NVIDIA Tesla V100.  `CUDA 10.0` and above should be used.  Training a single model should take up to three days, in this case, for most of the high-resolution benchmark datasets used in the literature.  With only two GPUs, about five days will be needed.  For older GPUs, especially those with less than 16 GB VRAM, modifications to the configuration files will be needed to ensure that out-of-memory issues are not encountered during training.  In particular, the batch sizes for the highest resolution should be lowered by up to half.


## MBRCE-AAE Results Generation

After training, there are several types of results that can be produced to illustrate how well the MBRCE-AAEs have modeled the data.

One of the claims that we make in our paper is that MBRCE-AAEs learn a posterior distribution that is equivalent to an entropy-augmented version of the training-set distribution.  The MBRCE-AAEs should thus be capable of producing realistic-looking samples.  This hypothesis can be tested by running `generate_mbrceaee_samples.py` and `generate_mbrceaee_generate_reconstructions.py`.  The former script randomly draws samples from the latent embedding space of the autoencoder and converts them into imagery by passing them through the MBRCE-AAE's decoder branch.  The latter script embeds the test samples, using the MBRCE-AAE's encoder branch, and then converts them back to imagery, using the MBRCE-AAE's decoder branch.

In the case of the CelebA dataset, you can show this via

    python3 generate_mbrceaee_samples.py -c ./configs/celeba-256x256.yaml
    python3 generate_mbrceaee_generate_reconstructions.py -c ./configs/celeba-256x256.yaml

This first script will produce a series of 1,024 facial images, which are located at `results/celeba-256x256-generated-samples/`.  Some examples are shown below, with the total set having an FID score of 13.72.

<p align="center">
  <img src="https://user-images.githubusercontent.com/32597777/154878202-3c77d7bc-7482-4851-a620-8a76b56a88c4.jpg" width="900"><br>
Example synthetically generated imagery for an MBRCE-AAE.  If the MBRCE-AAE has been trained well, then the generated images should have no perceptible flaws, such as misshapen faces, erroneous lighting conditions, and duplicated facial elements.  Here, we have trained the network on the CelebA face dataset at 256x256 resolution.
</p>

The second script will process all test-set imagery and store the results in `results/celeba-256x256-reconstructions/`.   Some example reconstructions are shown below.

<p align="center">
  <img src="https://user-images.githubusercontent.com/32597777/154879472-091581e0-e48d-46e7-913a-224f756f4ef4.jpg" width="900"><br>
Example reconstructed test-set imagery for an MBRCE-AAE.  The top row contains the input imagery while the bottom row contains the MBRCE-AAE's reconstruction response.  The goal of the network is to reduce discrepancies between its responses and the inputs while penalizing the synthetic responses that it generates.  If the MBRCE-AAE has been trained well, then the reconstructions should heavily resemble the inputs.  Here, we have trained the network on the CelebA face dataset at 256x256 resolution.
</p>

Due to how the MBRCE-AAEs are trained, they will not faithfully reproduce the training- and test-set imagery until very late in the training process.  However, even after only two hundred epochs, the networks have learned to compose human faces, model lighting, and integrate backgrounds well.

A reason why MBRCE-AAEs do well is that they heavily concentrate the embedding probability mass around the encoded training samples.  Due to this property, it may seem that they would only produce realistic results in local regions around the embedded training samples.  We have provided two scripts to demonstrate that this is not true, at least when the MBRCE-AAEs have been trained well.  The first `generate_mbrceaee_interp_samples.py` performs a multi-image interpolation process and displays the intermediate results.  This interpolation is conducted in the embedding space, with images being formed via the MBRCE-AAE's decoder branch.  The second, `generate_mbrceae_stylemix_samples.py` performs coarse-to-fine style mixing, transferring low-level and high-level attributes across image pairs.  This is done using a style-based generator network.

In the case of the CelebA dataset, running

    python3 generate_mbrceaee_interp_samples.py -c ./configs/celeba-256x256.yaml

produces the following result for images `00275.jpg` (top left corner), `00056.jpg` (top right corner), `00106.jpg` (bottom left corner), and `00123.jpg` (bottom right corner)

<p align="center">
  <img src="https://user-images.githubusercontent.com/32597777/154886282-af9162ba-e131-4dde-b0a1-8eb22a7115fe.jpg" width="900"><br>
Multi-image interpolation results for an MBRCE-AAE.  Each corner of the grid shows a different source image used to seed the interpolation.  The goal of the network is to generate intermediate facial images by linearly interpolating within the latent embedding space.  If the MBRCE-AAE has been trained well, then there should be a mostly smooth transition between any pair of faces in the grid.   Here, we have trained the network on the CelebA face dataset at 256x256 resolution.
</p>

Lastly, running

    python3 generate_mbrceae_stylemix_samples.py -c ./configs/celeba-256x256.yaml

can be used to produce the following results like the following for the images located in the `source` and `target` folders within `/style_mixing/test_images/celeba-256x256/`

<p align="center">
  <img src="https://user-images.githubusercontent.com/32597777/154885103-89758685-8e4b-4674-a340-823d2a5b252f.jpg" width="900"><br>
Example style transfer results for an MBRCE-AAE.  The goal is to transfer coarse facial features from the source images on the top row to the target images in the left column.  Here, we have trained the network on the CelebA face dataset at 256x256 resolution.
</p>

Here, we have only shown the coarse transfer results.

These results are the state of the art for adversarial autoencoder networks.  They are on roughly par with generative-adversarial networks from two to three years ago.  As we will show in an upcoming paper, the FID score can be nearly halved by integrating data augmentation processes within the stochastic game played by the network.  This substantially improves all aspects of the generated results presented above.


## Acknowledgements

Our MBRCE-AAE implementation draws heavily upon the work of:

S. Pidhorskyi et al., "Adversarial Latent Autoencoders", in Proceedings of the IEEE International Conference on Computer Vision and Pattern Recognition (CVPR), pp. 14104-14113, 2020. [DOI: 10.1109/CVPR42600.2020.01411] [arXiv: <a href="https://arxiv.org/abs/2004.04467">2004.04467</a>]

and 

T. Karras et al., "A Style-based Generator Architecture for Generative Adversarial Networks", in Proceedings of the IEEE International Conference on Computer Vision and Pattern Recognition (CVPR), pp. 4401-4410, 2019. [DOI: 10.1109/CVPR.2019.00453] [arXiv: <a href="https://arxiv.org/abs/1812.04948">1812.04948</a>]

T. Karras et al., "Progressive Growing of GANs for Improved Quality, Stability, and Variation", in Proceedings of the International Conference on Learning Representations (ICLR), pp. 1-26, 2018. [arXiv: <a href="https://arxiv.org/abs/1710.10196">1710.10196</a>]

and 

M. Lucic et al., "Are GANs Created Equal?  A Large-Scale Study". in Advances in Neural Information Processing Systems (NIPS), S. Bengio and H. Wallach and H. Larochelle and K. Grauman and N. Cesa-Bianchi and R. Garnett, Eds. Red Hook, NY, USA: Curran Associates, pp. 698–707, 2009. [arXiv: <a href="https://arxiv.org/abs/1711.10337">1711.10337</a>]
