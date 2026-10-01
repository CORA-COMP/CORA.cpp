// example_neuralNetwork_train_01_setBased - set-based training pushes decision boundaries off the samples
//
// The loss of set-based training (Koller, Ladner, Althoff: "Set-Based Training for Neural Network
// Verification", TMLR 2025) combines the usual loss of the center of the output set with the size of
// the output set, both for the input set of an eps-ball around every training point:
//
//     L = (1 - tau) CE(center of Y, t)  +  tau / eps * ||Y||_F,     ||Y||_F = sqrt(sum G^2) / n
//
// with the output zonotope Y = <c, G>. The gradients come from libtorch's autograd through the set
// propagation of `NeuralNetwork::evaluate`. Data and hyperparameters are those of the paper's Fig. 7
// (Appendix C): both networks learn the 20 points, but the decision boundary of the standard one
// cuts through the eps-boxes around the points, and that of the set-based one does not.
//
// Syntax:   build/examples/cpp/example_neuralNetwork_train_01_setBased [epochs, default 300]
// Outputs:  accuracy and certified share of the training points for both models
// See also: the Python example of the same name, example_neuralNetwork_verify_01

#include "nn/neuralNetwork/neuralNetwork.h"
#include "global/backend/torch.h"

#include <iostream>
#include <string>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

using namespace cora;

/// The four hidden layers of 100 ReLUs and the output layer of the paper's nn-med.
torch::nn::Sequential newModel() {
    torch::nn::Sequential model;
    int64_t in = 2;
    for (int layer = 0; layer < 4; ++layer) {
        model->push_back(torch::nn::Linear(in, 100));
        model->push_back(torch::nn::ReLU());
        in = 100;
    }
    model->push_back(torch::nn::Linear(100, 2));
    model->to(torch::kDouble);
    return model;
}

/// The linear layers of the model as the network of CORA.cpp (a bias is a column there).
NeuralNetwork asNetwork(const torch::nn::Sequential &model) {
    std::vector<NeuralNetwork::Layer> layers;
    for (const auto &module : model->children())
        if (auto linear = std::dynamic_pointer_cast<torch::nn::LinearImpl>(module))
            layers.push_back({fromTorch(linear->weight), fromTorch(linear->bias.unsqueeze(-1))});
    return NeuralNetwork(layers);
}

/// The output zonotopes of the eps-boxes around all points x (N, 2): one batch, one call.
Zonotope outputSet(const torch::nn::Sequential &model, const torch::Tensor &x, double eps) {
    const torch::Tensor generators =
        (eps * torch::eye(2, torch::kDouble)).expand({x.size(0), 2, 2}).contiguous();
    return asNetwork(model).evaluate(Zonotope(fromTorch(x.unsqueeze(-1)), fromTorch(generators)));
}

/// Per point whether every input of the eps-box gets its class: the largest value of
/// (logit of the other class - logit of the true class) over the output set is below 0.
torch::Tensor certified(const torch::nn::Sequential &model, const torch::Tensor &points,
                        const torch::Tensor &labels, double eps) {
    const torch::Tensor direction = torch::where(
        (labels == 0).unsqueeze(-1), torch::tensor({-1.0, 1.0}, torch::kDouble),
        torch::tensor({1.0, -1.0}, torch::kDouble));
    const Zonotope Y = outputSet(model, points, eps);
    return toTorch(Y.supportFunc(fromTorch(direction.unsqueeze(-1)))).flatten() < 0;
}

int main(int argc, char **argv) {
    setBackend("torch");
    torch::manual_seed(1);

    // Parameters ------------------------------------------------------------------------------

    // the Python example trains 1500 epochs; 300 already show the effect and take seconds
    const int epochs = argc > 1 ? std::stoi(argv[1]) : 300;
    const double epsTrain = 0.05;  // half-width of the input boxes during set-based training
    const double tau = 0.05;       // weight of the output-set size against the loss of the center
    const int64_t batchSize = 10;
    const double learningRate = 0.01;

    // Data ------------------------------------------------------------------------------------

    // the 20 points of the paper (Appendix C); the target (1, 0) is class 0, (0, 1) is class 1
    const torch::Tensor points = torch::tensor(
        {{0.0622, 0.6995}, {0.6534, 0.9409}, {0.4759, 0.7163}, {0.8812, 0.1020}, {0.5047, 0.4685},
         {0.1470, 0.3275}, {0.3439, 0.1395}, {0.9098, 0.5422}, {0.8588, 0.8696}, {0.0545, 0.0825},
         {0.6889, 0.4771}, {0.9329, 0.2857}, {0.6781, 0.3043}, {0.4641, 0.3302}, {0.4575, 0.9487},
         {0.1272, 0.4699}, {0.6506, 0.7315}, {0.5207, 0.1229}, {0.3271, 0.4574}, {0.6858, 0.0616}},
        torch::kDouble);
    const torch::Tensor labels = torch::tensor(
        {1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0}, torch::kLong);

    // Training --------------------------------------------------------------------------------

    const auto train = [&](bool setBased) {
        torch::nn::Sequential model = newModel();
        torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(learningRate));
        for (int epoch = 0; epoch < epochs; ++epoch) {
            const torch::Tensor order = torch::randperm(points.size(0), torch::kLong);
            for (int64_t start = 0; start < points.size(0); start += batchSize) {
                const torch::Tensor batch = order.narrow(0, start, batchSize);
                const torch::Tensor x = points.index_select(0, batch);
                const torch::Tensor t = labels.index_select(0, batch);
                optimizer.zero_grad();
                torch::Tensor loss;
                if (setBased) {
                    // the center of the output set and the F-radius of its generators (Prop. 5)
                    const Zonotope Y = outputSet(model, x, epsTrain);
                    const torch::Tensor G = toTorch(Y.G);
                    const torch::Tensor radius =
                        ((G * G).sum({-2, -1}) + 1e-12).sqrt() / static_cast<double>(G.size(-2));
                    loss = (1 - tau) * torch::nn::functional::cross_entropy(
                                           toTorch(Y.c).squeeze(-1), t) +
                           tau / epsTrain * radius.mean();
                } else {
                    loss = torch::nn::functional::cross_entropy(model->forward(x), t);
                }
                loss.backward();
                optimizer.step();
            }
        }
        return model;
    };

    // Evaluation ------------------------------------------------------------------------------

    std::cout << "method      accuracy   certified share of the 20 eps-boxes at eps = 0.02, 0.05\n";
    for (const bool setBased : {false, true}) {
        torch::nn::Sequential model = train(setBased);
        torch::NoGradGuard noGrad;
        const double accuracy =
            (model->forward(points).argmax(1) == labels).to(torch::kDouble).mean().item<double>();
        std::cout << (setBased ? "set-based" : "standard ") << "   " << accuracy;
        for (const double eps : {0.02, 0.05})
            std::cout << "   " << certified(model, points, labels, eps)
                                       .to(torch::kDouble).mean().item<double>();
        std::cout << "\n";
    }

    // example completed
    return 0;
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
