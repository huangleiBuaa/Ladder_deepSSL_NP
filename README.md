# Ladder network with Norm projection
The reimplementation of Ladder networks with norm projection. The codes are based on the original [torch implementation of Ladder network](https://github.com/joeyhng/ladder.torch). We add the projection based weight normalization method as introduced in the paper "projection based weight normalizaiton for deep neural netwroks" (update soon...). We achieved test errors as 2.52%, 1.06%, and 0.91% on Permunate invariant MNIST dataset (averaged over 10 random seeds) with only 20, 50, and 100 labeled samples respectively, which is the state-of-the-art results. 

You can reproduce the results with the scripts. Noting that the MNIST dataset (32x32 raw dataset) is required in the root dir with a path as `mnist.t7/train_32x32.t7' and `mnist.t7/test_32x32.t7' (you can change the path in the file 'MnistLoader.lua').
 


Single-time runs (meanstd normalization):

Dataset | network | test perf. |
--------|:-------:|:---------:|
CIFAR-10  | WRN-40-10-dropout | 3.8%
CIFAR-100 | WRN-40-10-dropout | 18.3%
SVHN      | WRN-16-8-dropout  | 1.54%



Considering the large variance of with 20 label training examples, we show the particular results of the 10 seeds, which respectively are 0.97%, 2.23%, 1.02%, 9.51%, 1.02%, 0.99%, 3.81%, 2.28%, 1.04%, 2.27%. 

