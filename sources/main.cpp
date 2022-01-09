#include <cstdio>
#include <memory>

#include "educnn.h"

enum { MLP_NETWORK_TYPE = 0, CNN_NETWORK_TYPE, NETWORK_TYPE_COUNT };

int main(int argc, char **argv) {
    int net_type = MLP_NETWORK_TYPE;
    if (argc > 1) {
        if (strcmp(argv[1], "--mlp") == 0) {
            net_type = MLP_NETWORK_TYPE;
        } else if (strcmp(argv[1], "--cnn") == 0) {
            net_type = CNN_NETWORK_TYPE;
        } else {
            fprintf(stderr, "Unknown flag \"%s\" is specified!", argv[1]);
            exit(1);
        }
    }

    // Parameters
    const int epochs = 6;
    const int batchsize = 64;

    // Train data
    Matrix train_data = mnist::train_images();
    Matrix train_labels = mnist::train_labels();
    const double eta = 1.0e-2;  // step size
    const double lambda = 0.1;  // momentum

    std::vector<std::shared_ptr<AbstractLayer>> layers;
    if (net_type == MLP_NETWORK_TYPE) {
        // MLP
        layers.emplace_back(new FullyConnectedLayer(784, 300));
        layers.emplace_back(new Sigmoid());
        layers.emplace_back(new FullyConnectedLayer(300, 10));
        layers.emplace_back(new LogSoftmax());
        printf("Network: MLP\n");
    } else if (net_type == CNN_NETWORK_TYPE) {
        // CNN
        layers.emplace_back(new ConvolutionLayer(Size(28, 28), Size(5, 5), 1, 6));
        layers.emplace_back(new MaxPoolingLayer(Size(24, 24), Size(2, 2), 6));
        layers.emplace_back(new ReLU());
        layers.emplace_back(new ConvolutionLayer(Size(12, 12), Size(5, 5), 6, 16));
        layers.emplace_back(new MaxPoolingLayer(Size(8, 8), Size(2, 2), 16));
        layers.emplace_back(new ReLU());
        layers.emplace_back(new FullyConnectedLayer(4 * 4 * 16, 84));
        layers.emplace_back(new ReLU());
        layers.emplace_back(new FullyConnectedLayer(84, 10));
        layers.emplace_back(new LogSoftmax());
        printf("Network: CNN\n");
    }
    Network network(layers);

    // Loss function
    auto criterion = std::make_shared<NLLLoss>();

    // Shuffle data indices
    Random &rng = Random::getInstance();
    const int n_data = (int)train_data.rows();

    Timer timer;
    timer.start();
    for (int e = 0; e < epochs; e++) {
        // Shuffle data indices
        std::vector<int> indices(n_data);
        for (int i = 0; i < n_data; i++) {
            indices[i] = i;
        }

        for (int i = 0; i < n_data; i++) {
            const int k = rng.nextInt(i, n_data - 1);
            std::swap(indices[i], indices[k]);
        }

        // Train with each batch
        const int n_batches = (n_data + batchsize - 1) / batchsize;
        ProgressBar pbar(n_batches);
        for (int j = 0; j < n_batches; j++) {
            // Construct a batch of data
            const int B = std::min(batchsize, n_data - j * batchsize);
            Matrix batch_data(B, train_data.cols());
            Matrix batch_labels(B, train_labels.cols());
            for (int b = 0; b < B; b++) {
                batch_data.row(b) = train_data.row(indices[j * batchsize + b]);
                batch_labels.row(b) = train_labels.row(indices[j * batchsize + b]);
            }

            // Process
            const Matrix &output = network.forward(batch_data);
            const Matrix &losses = criterion->forward(output, batch_labels);
            const double mean_loss = losses.mean();
            const double mean_acc = accuracy(output, batch_labels);

            network.backward(criterion->backward(), eta, lambda);
            pbar.setDescription("#%d: loss=%6.3f, acc=%6.3f", e + 1, mean_loss, mean_acc);
            pbar.step();
        }
    }

    printf("Time: %.2f sec\n", timer.stop());

    // Test
    Matrix test_data = mnist::test_images();
    Matrix test_labels = mnist::test_labels();

    Matrix pred = network.forward(test_data);
    const double acc = accuracy(pred, test_labels);
    printf("Acc: %6.2f %%\n", acc);
}
