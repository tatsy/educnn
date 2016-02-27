#include <cstdio>
#include <ctime>

#include "educnn.h"

int main(int argc, char** argv) {
    Matrix train_data = mnist::train_data();
    Matrix train_label = mnist::train_label();

    Random* rng = new Random((unsigned int)time(0));

    std::vector<AbstractLayer*> layers(2);
    layers[0] = new FullyConnectedLayer(rng, 768, 300);
    layers[1] = new FullyConnectedLayer(rng, 300, 10);

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
