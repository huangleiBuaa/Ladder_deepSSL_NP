# Ladder_deepSSL_NP
The reimplementation of Ladder networks with norm projection. The codes are based on the original [torch implementation of Ladder network](https://github.com/joeyhng/ladder.torch). We add the projection based weight normalization module as introduced in our paper "projection based weight normalizaiton for deep neural netwroks" (update soon...). We achieved test errors as $2.52\%$, $1.06\%$, and $0.91\%$ (averaged over 10 random seeds) with only 20, 50, and 100 labeled samples, respectively. You can reproduce the results with the scripts. Noting that the MNIST dataset (32x32 raw dataset) is required and put it in the root dir with a path as `mnist.t7/train_32x32.t7' and `mnist.t7/test_32x32.t7'


