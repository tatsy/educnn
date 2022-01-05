#include <cstdio>
#include <ctime>

#include "educnn.h"

int main(int argc, char** argv) {
    Matrix train_data = mnist::train_data();
    Matrix train_labels = mnist::train_labels();

    std::vector<AbstractLayer*> layers(4);
    layers[0] = new FullyConnectedLayer(784, 300);
    layers[1] = new Sigmoid();
    layers[2] = new FullyConnectedLayer(300, 10);
    layers[3] = new LogSoftmax();

    AbstractLoss *criterion = new NLLLoss();

    Network network(layers, criterion);

    Timer timer;
    timer.start();
    network.train(train_data, train_labels, 6, 128, 1.0e-2, 0.1);
    printf("Time: %.2f sec\n", timer.stop());

    Matrix test_data = mnist::test_data();
    Matrix test_labels = mnist::test_labels();

    Matrix pred = network.predict(test_data);

    const double acc = accuracy(pred, test_labels);
    printf("Acc: %6.2f %%\n", acc);

    for (int i = 0; i < layers.size(); i++) {
        delete layers[i];
    }
    delete criterion;
}
