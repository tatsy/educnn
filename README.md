educnn
===

> The simple implementation of CNN (convolutional neural network).

## Build & Run

For building the project, you need [Eigen](http://eigen.tuxfamily.org/index.php) (Version 3.0 or higher).

```shell
# Setup Eigen:
# (The version here is 3.2.7, but you can specify any versions higher than 3.0.0)
wget -O eigen-3.2.7.zip http://bitbucket.org/eigen/eigen/get/3.2.7.zip
unzip -qq eigen-3.2.7.zip -d $YOUR_EIGEN_DIR

# Build educnn
git clone https://github.com/tatsy/educnn.git
mkdir build && cd build
cmake -D EIGEN3_DIR=$YOUR_EIGEN_DIR ..

# Run
./bin/educnn
```

## Acknowledgments

The author sincerely appropriates for the following websites with fruitful information about deep learning and convolutional neural network.

* [Deep Learning Tutorial: Convolutional Neural Net (LeNet)](http://deeplearning.net/tutorial/lenet.html)
* [tiny-cnn](https://github.com/nyanp/tiny-cnn)

## Copyright

MIT License 2015-2022 (c) Tatsuya Yatagawa (tatsy)
