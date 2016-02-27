#include <cstdio>
#include <ctime>

#include "educnn.h"

int main(int argc, char** argv) {
    Matrix train_data = mnist::train_data();
    Matrix train_label = mnist::train_label();

    Random* rng = new Random((unsigned int)time(0));

    std::vector<AbstractLayer*> layers(6);
    layers[0] = new ConvolutionLayer(rng, Size(28, 28), Size(5, 5), 1, 4);
    layers[1] = new MaxPoolingLayer(rng, Size(24, 24), Size(2, 2), 4);
    layers[2] = new ConvolutionLayer(rng, Size(12, 12), Size(5, 5), 4, 6);
    layers[3] = new MaxPoolingLayer(rng, Size(8, 8), Size(2, 2), 6);
    layers[4] = new FullyConnectedLayer(rng, 96, 500);
    layers[5] = new FullyConnectedLayer(rng, 500, 10);

    Network network(layers, train_data, train_label, 50);

    Timer timer;
    timer.start();
    network.train(20, 0.1, 0.5);
    printf("Time: %f sec\n", timer.stop());

    Matrix test_data = mnist::test_data();
    Matrix test_label = mnist::test_label();

    Matrix pred = network.predict(test_data);

    double ratio = check(pred, test_label);
    printf("Ratio: %f %%\n", ratio);

    delete rng;
    for (int i = 0; i < layers.size(); i++) {
        delete layers[i];
    }
}
