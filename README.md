# image_generator_deepnetworks
Unsupervised Cross Domain Image Generation - Cross-domain Image Generation of a image
We studied the problem of transferring a image in one domain to another do-
main,assuming that both the domains here are related.We used Domain Transfer
Network(proposed by Fair Labs) and used a combination of loss functions which
includes multi-class GAN loss,a f-constancy loss ,and a regularizing loss.We tested
this network on image datasets of SVHN and MNIST.

To run the program ,run preprocess.py and then main.py with flag pretrain,train and then eval
