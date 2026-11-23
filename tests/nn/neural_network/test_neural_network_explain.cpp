// test_neural_network_explain - CORA's explain and computeFGSMAttack
//
// A linear network whose logit differences are exact on a box gives a hand-computed explanation:
// the least sensitive features are freed until the next one would flip the class. The abstract
// method with the original network at its only rate equals the standard one, a reduced network
// gives a freed set whose box the network classifies the same on samples, a flipped class is
// found by the gradient attack, and a regression network is judged by the radius of its output.
// The attack moves every unmasked feature by epsilon along the sign of the gradient. CORALean has
// no explanation, so these are property tests.
//
// CORA: nn/@neuralNetwork/explain.m
// CORA: nn/@neuralNetwork/computeFGSMAttack.m
// PROPERTY: nn/@neuralNetwork/explain.m
// PROPERTY: nn/@neuralNetwork/computeFGSMAttack.m
// CORA authors: Tobias Ladner
// Authors:       Claude Sonnet 5.5
// Written:       23-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/global/rng.h"
#include "cora/nn/neural_network/neural_network.h"
#include "testing.h"

#include <algorithm>
#include <cmath>
#include <functional>

using namespace cora;
using test::check;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Two classes, the first with the weights 3, 0.5, 0.1 on the inputs; the second always 0.
NeuralNetwork aux_linear() {
    return NeuralNetwork({{Tensor({{3.0, 0.5, 0.1}, {0.0, 0.0, 0.0}}), Tensor({0.0, 0.0})}},
                         {Activation::identity});
}

/// Hidden ReLU neurons h = relu(x), then the first class is a h0 - 2 h1 + 0.1 h2, the second 0.
NeuralNetwork aux_relu(double a = 1.0) {
    return NeuralNetwork(
        {{Tensor({{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}}), Tensor({0.0, 0.0, 0.0})},
         {Tensor({{a, -2.0, 0.1}, {0.0, 0.0, 0.0}}), Tensor({0.0, 0.0})}},
        {Activation::relu, Activation::identity});
}

/// Whether a sample of the box of the freed features (the others fixed) keeps the class.
bool aux_sound(const NeuralNetwork &nn, const Tensor &x, int64_t label, double eps,
               const std::vector<int64_t> &freed) {
    Rng rng(9);
    const int64_t n = nn.input_dim();
    std::vector<double> points;
    const std::vector<double> x0 = x.data();
    for (int k = 0; k < 200; ++k) {
        std::vector<double> u(static_cast<std::size_t>(n));
        rng.uniform(u.data(), u.size(), -eps, eps);
        for (int64_t i = 0; i < n; ++i)
            points.push_back(x0[std::size_t(i)] +
                             (std::find(freed.begin(), freed.end(), i) != freed.end()
                                  ? u[std::size_t(i)]
                                  : 0.0));
    }
    // no output beats the label's at any sample
    const std::vector<double> y = nn.evaluate(Tensor::from_data(points, {200, n})).data();
    const int64_t m = nn.output_dim();
    for (int k = 0; k < 200; ++k)
        for (int64_t j = 0; j < m; ++j)
            if (y[std::size_t(k * m + j)] > y[std::size_t(k * m + label)] + 1e-12) return false;
    return true;
}

void standard(const std::string &b) {
    const NeuralNetwork nn = aux_linear();
    const Tensor x({1.0, 1.0, 1.0});
    // the least sensitive first: 2 and 1 are freed, 0 would flip the class (3 (1 - 1.2) < 0)
    const Explanation e = nn.explain(x, 0, 1.2);
    check(e.feat_order == std::vector<int64_t>{2, 1, 0} && e.freed == std::vector<int64_t>{2, 1} &&
              e.times.size() == 3 && !std::isnan(e.times[0]) && aux_sound(nn, x, 0, 1.2, e.freed),
          b + ": the least sensitive features are freed");
    // all features at a small radius, with no time per feature; an explicit order
    ExplainOptions o;
    const Explanation all = nn.explain(x, 0, 0.1, o);
    check(all.freed.size() == 3 && all.times[2] == 0.0, b + ": all features can be freed");
    o.feat_order = {0, 1, 2};
    const Explanation ordered = nn.explain(x, 0, 1.3, o);
    check(ordered.freed == std::vector<int64_t>{1, 2} && ordered.times.size() == 3,
          b + ": feature 0 cannot go, the others can");
    // a timeout stops after the first feature
    o.timeout = 0.0;
    const Explanation cut = nn.explain(x, 0, 1.3, o);
    check(std::isnan(cut.times[1]) && !std::isnan(cut.times[0]), b + ": the timeout");
}

void regression(const std::string &b) {
    // y = 2 x0 + 0.1 x1: the radius of the output box is the sum of |weight| epsilon
    const NeuralNetwork nn({{Tensor({{2.0, 0.1}}), Tensor({0.0})}}, {Activation::identity});
    ExplainOptions o;
    o.output_threshold = 0.5;
    const Explanation e = nn.explain(Tensor({1.0, 1.0}), 0, 1.0, o);
    check(e.freed == std::vector<int64_t>{1} && e.feat_order == std::vector<int64_t>{1, 0},
          b + ": a regression network is judged by the radius");
}

void images(const std::string &b) {
    // two channels of a 1 x 2 image: a pixel frees both
    const NeuralNetwork nn(
        {{Tensor({{3.0, 0.1, 0.5, 0.0}, {0.0, 0.0, 0.0, 0.0}}), Tensor({0.0, 0.0})}},
        {Activation::identity});
    ExplainOptions o;
    o.input_size = {2, 1, 2};
    o.feat_order_method = "in-order";
    const Explanation e = nn.explain(Tensor({1.0, 1.0, 1.0, 1.0}), 0, 1.5, o);
    // pixel 0 is features 0 and 2 (3 x0 + 0.5 x2 + ...: negative), pixel 1 is features 1 and 3
    check(e.freed == std::vector<int64_t>{1} && e.feat_order == std::vector<int64_t>{0, 1},
          b + ": pixel 1 can be freed, pixel 0 cannot");
    o.feat_order = {0};
    check(nn.explain(Tensor({1.0, 1.0, 1.0, 1.0}), 0, 1.5, o).freed.empty(),
          b + ": pixel 0 frees both its channels or none");
}

void abstract(const std::string &b) {
    // the original network at its only rate is the standard method
    const NeuralNetwork nn = aux_relu();
    const Tensor x({1.0, 0.1, 0.0});
    ExplainOptions o;
    const Explanation s = nn.explain(x, 0, 0.5, o);
    o.method = "abstract+refine";
    o.refine_steps = {1.0};
    const Explanation a = nn.explain(x, 0, 0.5, o);
    check(a.freed == s.freed && a.feat_order == s.feat_order, b + ": rate 1 is the network");
    // feature 1 flips the class: relu(0.6) 2 > 1; the attack finds it, freeing 1 fails
    check(std::find(a.freed.begin(), a.freed.end(), 1) == a.freed.end() &&
              aux_sound(nn, x, 0, 0.5, a.freed),
          b + ": the flipping feature stays fixed");
    // reduced networks first, refined: sound and the same as the network on this example
    o.refine_steps = {0.5, 1.0};
    const Explanation r = nn.explain(x, 0, 0.5, o);
    check(aux_sound(nn, x, 0, 0.5, r.freed), b + ": reduced and refined");
    // a random refinement rises to the network, dynamic buckets
    o.refine_method = "rand";
    o.bucket_type = "dynamic";
    o.refine_steps = {0.5, 1.0};
    check(aux_sound(nn, x, 0, 0.5, nn.explain(x, 0, 0.5, o).freed), b + ": random refinement");
    // the default ladder 0.1 ... 1 and a ladder below 1
    o.refine_method = "all";
    o.refine_steps = {};
    check(aux_sound(nn, x, 0, 0.5, nn.explain(x, 0, 0.5, o).freed), b + ": the default ladder");
    o.refine_steps = {0.5};
    check(aux_sound(nn, x, 0, 0.5, nn.explain(x, 0, 0.5, o).freed), b + ": a ladder below 1");
    check(nn.explain(x, 0, 0.01, o).freed.size() == 3, b + ": all features at a tiny radius");
}

void attack(const std::string &b) {
    // the gradient of the loss of y = (3 x0 + 0.5 x1 + 0.1 x2, 0) at a point of class 0
    const NeuralNetwork nn = aux_linear();
    const Tensor x({{1.0, 1.0, 1.0}}), t({{1.0, 0.0}});
    const Tensor z = nn.compute_fgsm_attack(x, t, 0.25);
    // the loss falls as the first output rises, so the ascent moves every input down by epsilon
    check(test::close(z, Tensor({{0.75, 0.75, 0.75}}), 1e-12), b + ": every feature moves by eps");
    const Tensor masked = nn.compute_fgsm_attack(x, t, 0.25, Tensor({1.0, 0.0, 1.0}));
    check(test::close(masked, Tensor({{0.75, 1.0, 0.75}}), 1e-12), b + ": a masked feature stays");
    // the shapes are checked
    check(test::raises([&] { nn.compute_fgsm_attack(x, Tensor({{1.0}}), 0.25); },
                       Error::dimension_mismatch),
          b + ": a wrong target");
    check(test::raises([&] { nn.compute_fgsm_attack(x, t, 0.25, Tensor({1.0})); },
                       Error::dimension_mismatch),
          b + ": a wrong mask");
}

void errors(const std::string &b) {
    const NeuralNetwork nn = aux_linear();
    const Tensor x({1.0, 1.0, 1.0});
    const auto bad = [&](const std::function<void(ExplainOptions &)> &set, Error::Id e) {
        ExplainOptions o;
        set(o);
        return test::raises([&] { nn.explain(x, 0, 0.5, o); }, e);
    };
    // the point, the label and the radius
    check(test::raises([&] { nn.explain(Tensor({1.0}), 0, 0.5); }, Error::dimension_mismatch),
          b + ": a wrong point");
    check(test::raises([&] { nn.explain(x, 2, 0.5); }, Error::wrong_value), b + ": a wrong label");
    check(test::raises([&] { nn.explain(x, 0, -1.0); }, Error::wrong_value),
          b + ": a negative epsilon");
    // the options
    check(bad([](ExplainOptions &o) { o.method = "x"; }, Error::wrong_value), b + ": method");
    check(bad([](ExplainOptions &o) { o.refine_method = "x"; }, Error::wrong_value),
          b + ": refine method");
    check(bad(
              [](ExplainOptions &o) {
                  o.method = "abstract+refine";
                  o.refine_method = "sensitivity";
              },
              Error::not_supported),
          b + ": refinement by sensitivity");
    check(bad([](ExplainOptions &o) { o.bucket_type = "x"; }, Error::wrong_value), b + ": buckets");
    check(bad([](ExplainOptions &o) { o.refine_steps = {0.5, 0.2}; }, Error::wrong_value),
          b + ": steps");
    check(bad([](ExplainOptions &o) { o.input_size = {1, 2, 2}; }, Error::dimension_mismatch),
          b + ": input size");
    check(bad([](ExplainOptions &o) { o.feat_order = {5}; }, Error::wrong_value),
          b + ": a feature outside the input");
    check(bad([](ExplainOptions &o) { o.feat_order_method = "snake"; }, Error::wrong_value),
          b + ": a snake needs the size");
    // an activation the reduction does not take
    const NeuralNetwork soft({{Tensor({{3.0, 0.5, 0.1}, {0.0, 0.0, 0.0}}), Tensor({0.0, 0.0})}},
                             {Activation::softplus});
    ExplainOptions ar;
    ar.method = "abstract+refine";
    check(test::raises([&] { soft.explain(x, 0, 0.5, ar); }, Error::not_supported),
          b + ": an activation the reduction does not take");
    // a gather layer is made dense: the same explanation as the matrix
    const NeuralNetwork gathered(
        {NeuralNetwork::Layer::gather({0, 1, 2}, 3, Tensor::zeros({3})),
         NeuralNetwork::Layer{Tensor({{3.0, 0.5, 0.1}, {0.0, 0.0, 0.0}}), Tensor({0.0, 0.0})}},
        {Activation::identity, Activation::identity});
    check(gathered.explain(x, 0, 1.2).freed == std::vector<int64_t>{2, 1}, b + ": a gather layer");
    {
        // a valid call without checks
        ChecksScope off(false);
        check(!nn.explain(x, 0, 1.2).freed.empty(), b + ": a valid call without checks");
    }
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        standard(b);
        regression(b);
        images(b);
        abstract(b);
        attack(b);
        errors(b);
    });
    return test::finish("neuralNetwork explain");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
