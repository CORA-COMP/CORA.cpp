// test_neural_network_region_polys - the merge buckets and the polynomials region by region
//
// The buckets are the finite limits of the activation; the regions of find_region_polys follow
// each other from -inf to inf, each with an error below the tolerance, and their polynomials
// enclose the activation within their error. Property: on random tolerances and domains (fixed
// seed), every region is within the tolerance and encloses the activation, and a region whose
// constant fit is within the tolerance takes the constant (CORA lowers the order down to 0).
//
// CORA: nn/layers/nonlinear/@nnActivationLayer/getMergeBuckets.m
// CORA: nn/layers/nonlinear/nnReLULayer.m
// CORA: nn/layers/nonlinear/@nnActivationLayer/findRegionPolys.m
// PROPERTY: nn/layers/nonlinear/@nnActivationLayer/findRegionPolys.m
// CORA authors: Tobias Ladner, Lukas Koller
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5
// Written:       23-November-2026
// Last update:   25-November-2026
// Reviewed:      ---

#include "cora/global/rng.h"
#include "cora/nn/neural_network/neural_network.h"
#include "nn/neural_network/private/priv_activation_poly.h"
#include "testing.h"

#include <algorithm>
#include <cmath>
#include <limits>

using namespace cora;
using test::check;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

void buckets(const std::string &b) {
    using V = std::vector<double>;
    check(NeuralNetwork::merge_buckets(Activation::relu) == V{0.0}, b + ": relu");
    check(NeuralNetwork::merge_buckets(Activation::sigmoid) == V{0.0, 1.0}, b + ": sigmoid");
    check(NeuralNetwork::merge_buckets(Activation::tanh) == V{-1.0, 1.0}, b + ": tanh");
    check(NeuralNetwork::merge_buckets(Activation::softplus) == V{0.0}, b + ": softplus");
    check(NeuralNetwork::merge_buckets(Activation::identity).empty(), b + ": identity");
    check(NeuralNetwork::merge_buckets(Activation::root).empty(), b + ": root (NaN, inf)");
    check(NeuralNetwork::merge_buckets(Activation::reciprocal) == V{1.0, 0.0}, b + ": reciprocal");
    check(test::raises([] { NeuralNetwork::merge_buckets(Activation::group_sort); },
                       Error::not_supported),
          b + ": a group sort has no function of one neuron");
}

void regions(const std::string &b) {
    const double tol = 0.05, inf = std::numeric_limits<double>::infinity();
    const std::vector<RegionPoly> R = NeuralNetwork::find_region_polys(
        Activation::sigmoid, tol, 2, -6.0, 6.0, {0.0}, 0.0025, {1.0}, 0.0025);
    check(R.size() >= 3 && R.front().l == -inf && R.back().u == inf, b + ": from -inf to inf");
    // contiguous regions, each within the tolerance and enclosing the sigmoid
    bool contiguous = true, within = true, encloses = true;
    for (std::size_t i = 0; i < R.size(); ++i) {
        if (i > 0) contiguous = contiguous && R[i].l == R[i - 1].u;
        within = within && R[i].d <= tol + 1e-12;
        const double lo = std::max(R[i].l, -8.0), hi = std::min(R[i].u, 8.0);
        for (int j = 0; j <= 100 && lo < hi; ++j) {
            const double x = lo + (hi - lo) * double(j) / 100.0;
            encloses = encloses && std::abs(1.0 / (1.0 + std::exp(-x)) - priv_polyval(R[i].p, x)) <=
                                       R[i].d + 1e-12;
        }
    }
    check(contiguous && within && encloses, b + ": contiguous regions of error below tol");
    // a smaller tolerance needs more regions
    const std::vector<RegionPoly> finer = NeuralNetwork::find_region_polys(
        Activation::sigmoid, 0.01, 2, -6.0, 6.0, {0.0}, 0.0025, {1.0}, 0.0025);
    check(finer.size() > R.size(), b + ": a finer tolerance, more regions");
    // an outer radius above the tolerance warns, and the loop is silent about it
    int warnings = 0;
    const WarningHandler previous =
        set_warning_handler([&](const std::string &, const std::string &) { ++warnings; });
    NeuralNetwork::find_region_polys(Activation::sigmoid, tol, 1, -3.0, 3.0, {0.0}, 0.2, {1.0},
                                     0.2);
    set_warning_handler(previous);
    check(warnings == 2, b + ": the outer radii above tol warn");
    check(test::raises(
              [] {
                  NeuralNetwork::find_region_polys(Activation::sigmoid, 0.0, 1, -3.0, 3.0, {0.0},
                                                   0.0, {1.0}, 0.0);
              },
              Error::wrong_value),
          b + ": a tolerance of 0");
    {
        // a valid call without checks
        ChecksScope off(false);
        check(NeuralNetwork::find_region_polys(Activation::tanh, 0.2, 1, -2.0, 2.0, {-1.0}, 0.0,
                                               {1.0}, 0.0)
                      .size() >= 3,
              b + ": a valid call without checks");
    }
}

/// Random tolerances and domains of the sigmoid and tanh: within tol, enclosing, constants where
/// a constant suffices (the saturated tails at a loose tolerance).
void property(const std::string &b) {
    Rng rng(11);
    bool within = true, encloses = true, constant = false;
    for (int rep = 0; rep < 6; ++rep) {
        const Activation a = rep % 2 == 0 ? Activation::sigmoid : Activation::tanh;
        double tol = 0.0, e = 0.0;
        rng.uniform(&tol, 1, 0.01, 0.08);
        rng.uniform(&e, 1, 3.0, 8.0);
        const double lo_lim = a == Activation::tanh ? -1.0 : 0.0;
        const std::vector<RegionPoly> R =
            NeuralNetwork::find_region_polys(a, tol, 2, -e, e, {lo_lim}, 0.0, {1.0}, 0.0);
        for (const RegionPoly &r : R) {
            // the outer regions are the polynomials given
            if (!std::isfinite(r.l) || !std::isfinite(r.u)) continue;
            within = within && r.d <= tol;
            constant = constant || r.p.size() == 1;
            for (int j = 0; j <= 50; ++j) {
                const double x = r.l + (r.u - r.l) * double(j) / 50.0;
                const double f = a == Activation::tanh ? std::tanh(x) : 1.0 / (1.0 + std::exp(-x));
                encloses = encloses && std::abs(f - priv_polyval(r.p, x)) <= r.d + 1e-12;
            }
        }
    }
    check(within && encloses, b + ": random tolerances, every region within tol and enclosing");
    check(constant, b + ": a saturated region takes the constant fit");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        buckets(b);
        regions(b);
        property(b);
    });
    return test::finish("neuralNetwork region polys");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
