# Ladder network with Norm projection
The reimplementation of Ladder networks with norm projection. The codes are based on the original [torch implementation of Ladder network](https://github.com/joeyhng/ladder.torch). We add the projection based weight normalization method as introduced in the paper "projection based weight normalizaiton for deep neural netwroks" (update soon...). We achieved test errors as 2.52%, 1.06%, and 0.91% on Permunate invariant MNIST dataset (averaged over 10 random seeds) with only 20, 50, and 100 labeled samples respectively, which is the state-of-the-art results. 

You can reproduce the results with the scripts. Noting that the MNIST dataset (32x32 raw dataset) is required in the root dir with a path as `mnist.t7/train_32x32.t7' and `mnist.t7/test_32x32.t7' (you can change the path in the file 'MnistLoader.lua').
 


Comparison of test errors (%) for semi-supervised setup on permutation invariant MNIST dataset. We show the
test error for a given number of samples={20,50,100} with a form of mean(+- std). Ladder* indicates our implementation of Ladder network [1].

method | 20 labeled  | 50 labeled | 100 labeled |
--------|:-------:|:---------:|:---------:|
Auxiliary Deep Generative Model [2] | - | - |0.96 ± 0.02
Virtual Adversarial [3]            | - | - |1.36
Ladder [1]                         | - | 1.62 ± 0.65 |1.06 ± 0.37
Ladder+AMLP [4]                    | - | - |1.002 ± 0.038
Improved GANs with feature matching [5]      | 16.77 ± 4.52 | 2.21 ± 1.36 |0.93 ± 0.065
Triple-GAN [6]                     | 4.81 ± 4.95 | 1.56 ± 0.72 |0.91 ± 0.58
--------|:-------:|:---------:|:---------:|
Ladder* (our implementation)        |9.67 ± 10.1 | 3.53 ± 6.6 | 1.12 ± 0.59 
Ladder+PBWN (ours)                  |2.52 ± 2.42 | 1.06 ± 0.48| 0.91 ± 0.05



Considering the large variance of 10 seeds with 20 label training examples , we show the particular results of the 10 seeds of our method, which respectively are 0.97%, 2.23%, 1.02%, 9.51%, 1.02%, 0.99%, 3.81%, 2.28%, 1.04%, 2.27%. 



## Reference
[1] Antti Rasmus, Harri Valpola, Mikko Honkala, Mathias Berglund, and Tapani Raiko. Semi-supervised learning with ladder networks. In NIPS, 2015.
[2] Lars Maale, Casper Kaae Snderby, Sren Kaae Snderby, and Ole Winther. Auxiliary deep generative models. In ICML, 2016
[3] Takeru Miyato, Shin-ichi Maeda, Masanori Koyama, and Shin Ishii. Virtual adversarial training: a regularization method for supervised and semi-supervised learning. CoRR, abs/1704.03976, 2017.
[4] Mohammad Pezeshki, Linxi Fan, Philemon Brakel, Aaron C. Courville, and Yoshua Bengio. Deconstructing the ladder network architecture. In ICML, 2016.
[5] Tim Salimans, Ian J. Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, and Xi Chen. Improved techniques for training gans. In NIPS, pages 2226–2234, 2016.
[6] Chongxuan Li, Kun Xu, Jun Zhu, and Bo Zhang. Triple generative adversarial nets. CoRR, abs/1703.02291, 2017.
## Contact
huanglei@nlsde.buaa.edu.cn, Any discussions and suggestions are welcome

