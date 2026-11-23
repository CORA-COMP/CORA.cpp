// test_lean_nn_relaxation - the lines of NeuralNetwork::verify's activations against CORALean
//
// The line of sign (priv_sign_line) and the square's chord-tangent band (priv_square_line) over
// [l, u], bit for bit against the oracle's nnLayer.relaxation (signRelaxation,
// squareChordTangentRelaxation: CORA.cpp's formula, every rounding cost zero at nearest), on boxes
// below, above and across zero, points, and boxes from 1e-3 to 1e3 wide.
//
// CORA.cpp only: tests priv_sign_line and priv_square_line (CORA has no zonotope line of sign or of
// the square)
// Authors:       Claude Opus 5.5
// Written:       23-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/global/oracle/neural_network.h"
#include "nn/neural_network/private/priv_layers.h"
#include "oracle_testing.h"

#include <cmath>

using namespace cora;

namespace oracle = test::oracle;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Boxes [l, u] as columns (n, 1): random ones of every scale, plus points and boxes at zero.
std::pair<Tensor, Tensor> aux_boxes(oracle::Random &rnd, int64_t n) {
    std::vector<double> l, u;
    for (int64_t i = 0; i < n; ++i) {
        const double scale = std::pow(10.0, double(i % 7) - 3.0);
        const double a = rnd.scalar() * scale, b = rnd.scalar() * scale;
        l.push_back(std::min(a, b));
        u.push_back(std::max(a, b));
    }
    // a point, a box ending at zero, one starting at zero
    l.insert(l.end(), {0.75, -2.0, 0.0});
    u.insert(u.end(), {0.75, 0.0, 3.0});
    const auto m = int64_t(l.size());
    return {Tensor::from_data(l, {m, 1}), Tensor::from_data(u, {m, 1})};
}

/// The line against the oracle's slope, offset and error.
void aux_check(const std::string &kind, const PrivLine &line, const lean::NnLine &lean) {
    oracle::check(kind + " slope", line.slope, lean.slope);
    oracle::check(kind + " offset", line.offset, lean.offset);
    oracle::check(kind + " radius", line.radius, lean.error);
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    if (!oracle::available() || !oracle::serves("nnLayer.relaxation")) return 0;
    oracle::run("nnLayer.relaxation", [&](oracle::Random &rnd) {
        for (const int64_t n : {1, 6, 40}) {
            const auto [l, u] = aux_boxes(rnd, n);
            aux_check("sign", priv_sign_line(l, u), lean::relaxation("sign", l, u));
            aux_check("square", priv_square_line(l, u), lean::relaxation("square", l, u));
        }
    });
    return test::finish("lean nn relaxation");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
