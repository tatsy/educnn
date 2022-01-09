educnn
===

[![Windows CI](https://github.com/tatsy/educnn/actions/workflows/windows.yaml/badge.svg)](https://github.com/tatsy/educnn/actions/workflows/windows.yaml)
[![MacOS CI](https://github.com/tatsy/educnn/actions/workflows/macos.yaml/badge.svg)](https://github.com/tatsy/educnn/actions/workflows/macos.yaml)
[![Ubuntu CI](https://github.com/tatsy/educnn/actions/workflows/ubuntu.yaml/badge.svg)](https://github.com/tatsy/educnn/actions/workflows/ubuntu.yaml)

> C++ implementation of convolutional neural network with a plenty of comments for easy understanding.  
> C++による畳み込みニューラルネットの実装 (コード理解のためのコメント付き)

## Build & Run

For building the project, you need [Eigen](http://eigen.tuxfamily.org/index.php) (Version 3.0 or higher).  
このプロジェクトのビルドのためには予め[Eigen](http://eigen.tuxfamily.org/index.php)をインストールしてください (バージョン3以上)

```shell
# Setup Eigen (The version here is 3.2.7, but you can specify any versions higher than 3.0.0)
# Eigenのセットアップ (ここではバージョン3.2.7を指定していますが、3.0.0以上なら別のバージョンでも構いません)
wget -O eigen-3.2.7.zip http://bitbucket.org/eigen/eigen/get/3.2.7.zip
unzip -qq eigen-3.2.7.zip -d $YOUR_EIGEN_DIR

# Build educnn
# educnnのビルド
git clone https://github.com/tatsy/educnn.git
mkdir build && cd build
cmake -D EIGEN3_DIR=$YOUR_EIGEN_DIR ..

# Run educnn
# プログラムの実行
./bin/educnn
```

## Acknowledgments

The author sincerely appropriates for the following websites with fruitful information about deep learning and convolutional neural network.

* [Deep Learning Tutorial: Convolutional Neural Net (LeNet)](http://deeplearning.net/tutorial/lenet.html)
* [tiny-cnn](https://github.com/nyanp/tiny-cnn)

## Copyright

MIT License 2015-2022 (c) Tatsuya Yatagawa (tatsy)
