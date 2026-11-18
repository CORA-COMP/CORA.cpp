// test_polytope_linear_programs - Polytope::LinearPrograms: programs over a polytope in a box
//
// A segment of the plane in the unit box and a halfspace in a box, by hand: emptiness, the range
// of a cost, the maximum and its point, the bounds of the coordinates and the Chebyshev center.
// On random constrained zonotopes the programs of the factor polytope give the same bits as the
// linear program with the box as bounds (the stated property: the box stays bounds).
//
// CORA.cpp only: tests LinearPrograms, the programs CORA's conZonotope hands to CORAlinprog
// Authors:       Claude Opus 5.5
// Written:       17-November-2026
// Last update:   18-November-2026
// Reviewed:      ---

#include "cora/cont_set/con_zonotope/con_zonotope.h"
#include "cora/global/linprog.h"
#include "polytope_testing.h"

#include <cmath>
#include <limits>

using namespace cora;
using test::check;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

/// The line x1 + x2 = 1 in the unit box [-1, 1]^2: the segment from (0, 1) to (1, 0).
void segment(const std::string &b) {
    const Polytope line(Tensor::zeros({0, 2}), Tensor::zeros({0, 1}), Tensor({{1.0, 1.0}}),
                        column({1.0}));
    const Polytope::LinearPrograms lp = line.linear_programs({-1.0, -1.0}, {1.0, 1.0});
    check(!lp.is_empty(), b + ": the segment has a point");
    const auto r = lp.range({1.0, -1.0});
    check(r && std::abs(r->first + 1.0) < 1e-12 && std::abs(r->second - 1.0) < 1e-12,
          b + ": x1 - x2 ranges over [-1, 1]");
    const auto best = lp.maximum({1.0, 0.0});
    check(best && std::abs(best->value - 1.0) < 1e-12 && std::abs(best->x[1]) < 1e-12,
          b + ": x1 is largest at (1, 0)");
    const auto [lo, hi] = lp.bounds();
    check(std::abs(lo[0]) < 1e-12 && std::abs(hi[0] - 1.0) < 1e-12 && std::abs(lo[1]) < 1e-12,
          b + ": both coordinates range over [0, 1]");
    const auto x = lp.chebyshev_center();
    check(x && std::abs((*x)[0] + (*x)[1] - 1.0) < 1e-12, b + ": the center is on the line");
    // the line x1 + x2 = 3 misses the box
    const Polytope far(Tensor::zeros({0, 2}), Tensor::zeros({0, 1}), Tensor({{1.0, 1.0}}),
                       column({3.0}));
    const Polytope::LinearPrograms none = far.linear_programs({-1.0, -1.0}, {1.0, 1.0});
    check(none.is_empty() && !none.range({1.0, 0.0}) && !none.maximum({1.0, 0.0}),
          b + ": no point, no range and no maximum");
    check(!none.chebyshev_center(), b + ": no center");
}

/// The halfspace x1 <= 0 in the box [-2, 2] x [-1, 1], and without a box (infinite bounds).
void halfspace(const std::string &b) {
    const Polytope H(Tensor({{1.0, 0.0}}), column({0.0}));
    const Polytope::LinearPrograms lp = H.linear_programs({-2.0, -1.0}, {2.0, 1.0});
    const auto [lo, hi] = lp.bounds();
    check(std::abs(lo[0] + 2.0) < 1e-12 && std::abs(hi[0]) < 1e-12 &&
              std::abs(lo[1] + 1.0) < 1e-12 && std::abs(hi[1] - 1.0) < 1e-12,
          b + ": the bounds are the box cut at 0");
    const auto x = lp.chebyshev_center();
    check(x && std::abs((*x)[0] + 1.0) < 1e-12 && std::abs((*x)[1]) < 1e-12,
          b + ": the ball of radius 1 centered at (-1, 0)");
    const Polytope::LinearPrograms free = H.linear_programs({-kInf, -kInf}, {kInf, kInf});
    check(!free.is_empty() && !free.range({1.0, 0.0}), b + ": unbounded below, no range");
}

/// The programs of the factor polytope of random sets: the bits of the program with the box as
/// bounds.
void same_programs() {
    Rng rng(7);
    for (int k = 0; k < 10; ++k) {
        const ConZonotope cz = ConZonotope::generate_random(rng, 2 + k % 3);
        const int64_t m = cz.nr_generators();
        const Polytope::LinearPrograms lp = cz.factor_programs();
        LinProg direct;
        direct.n = int(m);
        direct.Aeq = cz.A().shape()[0] > 0 ? cz.A().data() : std::vector<double>{};
        direct.beq = cz.b().shape()[0] > 0 ? cz.b().data() : std::vector<double>{};
        direct.lb.assign(std::size_t(m), -1.0);
        direct.ub.assign(std::size_t(m), 1.0);
        std::vector<double> f(static_cast<std::size_t>(m));
        rng.uniform(f.data(), f.size(), -0.5, 0.5);
        direct.f = f;
        for (double &x : direct.f) x = -x;
        // the program with the box as bounds, solved directly
        const LinProgResult r = linprog(direct);
        const auto best = lp.maximum(f);
        check(r.status == LinProgResult::Status::optimal && best && best->value == -r.fval &&
                  best->x == r.x,
              "the maximum of the factor programs is the program's, bit for bit (" +
                  std::to_string(k) + ")");
    }
}

/// A batch and a box of the wrong size are refused, the batch also with the checks off.
void refusals() {
    const Polytope line(Tensor::zeros({0, 2}), Tensor::zeros({0, 1}), Tensor({{1.0, 1.0}}),
                        column({1.0}));
    check(test::raises([&] { line.linear_programs({-1.0}, {1.0}); }, Error::dimension_mismatch),
          "a box of one dimension for two");
    const Polytope batch(Tensor::ones({2, 1, 2}), Tensor::ones({2, 1, 1}));
    ChecksScope off(false);
    check(test::raises([&] { batch.linear_programs({-1.0, -1.0}, {1.0, 1.0}); },
                       Error::not_supported),
          "a batch, with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        segment(b);
        halfspace(b);
    });
    same_programs();
    refusals();
    return test::finish("polytope linear programs");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
