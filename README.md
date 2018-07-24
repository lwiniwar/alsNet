## alsNet: Classification of 3D Point Clouds using Deep Neural Networks

This is the code repository accompanying the diploma thesis that was 
carried out at the <a href="photo.geo.tuwien.ac.at">Research Group Photogrammetry</a> of TU Wien 
under the same name.

*alsNet* is a neural network framework for classification of point clouds acquired by airborne laser scanning.
More details can be found in the thesis itself.

### PointNet, PointNet++
*alsNet* is heavily based on the neural networks of *PointNet* and *PointNet++* by Charles R. Qi et al. from Stanford University.
This especially concerns the tensorflow operations. 
The code has been updated to run on tensorflow 1.6, CUDA 9.0 and python3. To complile them, the instructions below can be follow. They are copied from the *PointNet++* repository.
Since some changes have been made to the `tf_xxx_compile.sh` scripts, they should run as-is, provided a correct installation of CUDA, cuDNN and tensorflow-gpu exists.
#### Compile Customized TF Operators
The TF operators are included under `tf_ops`, you need to compile them (check `tf_xxx_compile.sh` under each ops subfolder) first. Update `nvcc` and `python` path if necessary. The code is tested under TF1.2.0. If you are using earlier version it's possible that you need to remove the `-D_GLIBCXX_USE_CXX11_ABI=0` flag in g++ command in order to compile correctly.

To compile the operators in TF version >=1.4, you need to modify the compile scripts slightly.

First, find Tensorflow include and library paths.

        TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
        TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
        
Then, add flags of `-I$TF_INC/external/nsync/public -L$TF_LIB -ltensorflow_framework` to the `g++` commands.


### Usage
This sections shows how to use *alsNet* both in training and in inference. With all of these scripts, the parameter `-h` will show help information on the other parameters.

#### Preprocessing
First, the dataset has to be tiled into chunks of 200,000 points each. Here we take a number of laz-Files, do not thin them out (`thinFactor 1`) 
and assume an average point density of 15 pts/m².

    alsNet/alsNet/alsNetPreparer.py --inFiles .../input*.laz --outFolder .../some/folder --thinFactor 1 --density 15 --kNN 200000

#### Training
Now we can train a model based on these chunks and an architecture (e.g. `arch4`)

    alsNet/alsNet/alsNetRunner5.py --inList .../some/folder/stats.csv --threshold 20 --minBuild 0 --learningRate 0.0001 --outDir .../logs_models/ --archFile archs.arch4

If we want to continue training on an exisiting model, we can supply the path to the already saved files: `--continueModel .../logs_models/model_1_99.alsNet`

#### Inferece
To use a trained model on validation data, use the `alsNetEvaluator`. The data has to be prepared using the `alsNetPreparer` in advance.

    alsNet/alsNet/alsNetEvaluator.py --inFile .../data/validate_c*.laz --model .../logs_models/model_1_99.alsNet --arch archs.arch4 --outDir .../predictions/

#### Postprocessing
Finally, the chunks can be merged together to create a single output file:

    alsNet/alsNet/alsNetMerger.py 

### License
The code is released under MIT License (see LICENSE file for details).

### Related Projects

* <a href="http://stanford.edu/~rqi/pointnet" target="_blank">PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation</a> by Qi et al. (CVPR 2017 Oral Presentation). Code and data released in <a href="https://github.com/charlesq34/pointnet">GitHub</a>.
* <a href="http://stanford.edu/~rqi/pointnet2/" target="_blank">PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space</a> by Qi et al. (NIPS 2017) A hierarchical feature learning framework on point clouds. The PointNet++ architecture applies PointNet recursively on a nested partitioning of the input point set. It also proposes novel layers for point clouds with non-uniform densities.