#include <cstdio>
#include <ctime>

#include "educnn.h"

int main(int argc, char** argv) {
    Matrix train_data = mnist::train_data();
    Matrix train_labels = mnist::train_labels();

    std::vector<AbstractLayer*> layers(10);
    layers[0] = new ConvolutionLayer(Size(28, 28), Size(5, 5), 1, 6);
    layers[1] = new MaxPoolingLayer(Size(24, 24), Size(2, 2), 6);
    layers[2] = new ReLU();
    layers[3] = new ConvolutionLayer(Size(12, 12), Size(5, 5), 6, 16);
    layers[4] = new MaxPoolingLayer(Size(8, 8), Size(2, 2), 16);
    layers[5] = new ReLU();
    layers[6] = new FullyConnectedLayer(4 * 4 * 16, 84);
    layers[7] = new ReLU();
    layers[8] = new FullyConnectedLayer(84, 10);
    layers[9] = new LogSoftmax();

    AbstractLoss *criterion = new NLLLoss();

    Network network(layers, criterion);
    
    Timer timer;
    timer.start();
    network.train(train_data, train_labels, 6, 128, 1.0e-2, 0.5);
    printf("Time: %f sec\n", timer.stop());

    Matrix test_data = mnist::test_data();
    Matrix test_labels = mnist::test_labels();

    Matrix pred = network.predict(test_data);

    double ratio = check(pred, test_labels);
    printf("Ratio: %f %%\n", ratio);

    for (int i = 0; i < layers.size(); i++) {
        delete layers[i];
    }
    delete criterion;
}
