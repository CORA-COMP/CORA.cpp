// test_interval_quad_map - interval quad_map: the box around a quadratic map of boxes
//
// CORA: contSet/@interval/quadMap.m
// PROPERTY: contSet/@interval/quadMap.m

#include "cora/cont_set/interval/interval.h"
#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/rng.h"
#include "testing.h"

#include <limits>

using namespace cora;
using test::check;
using test::close;
using test::column;

namespace {

/// 2 x^2 on [-1, 1] is enclosed by [0, 2] (exact here); x1 x2 on [1, 2] x [-1, 3] by [-3, 6]
/// (CORA's zonotope map: center 1.5, generators 3, 0.5, 1).
void quad_map_by_hand(const std::string &b) {
    const Interval I(column({-1.0}), column({1.0}));
    const Interval R = I.quad_map({Tensor({{2.0}})});
    check(close(R.inf(), std::vector<double>{0}, 1e-14) &&
              close(R.sup(), std::vector<double>{2}, 1e-14),
          b + ": 2 x^2 on [-1, 1]");
    const Interval X(column({1.0}), column({2.0})), Y(column({-1.0}), column({3.0}));
    const Interval M = X.quad_map(Y, {Tensor({{1.0}})});
    check(close(M.inf(), std::vector<double>{-3}, 1e-14) &&
              close(M.sup(), std::vector<double>{6}, 1e-14),
          b + ": x1 x2 on [1, 2] x [-1, 3]");
}

/// It is the interval hull of Zonotope::quad_map of the zonotopes of the boxes, and encloses the
/// map of sampled points.
void quad_map_is_the_hull_of_the_zonotope_map(const std::string &b) {
    cora::Rng rng(141);
    const Interval I = Interval::generate_random(rng, 3), J = Interval::generate_random(rng, 3);
    std::vector<Tensor> Q;
    for (int i = 0; i < 2; ++i) {
        std::vector<double> data(9);
        rng.normal(data.data(), data.size(), 1.0);
        Q.push_back(Tensor::from_data(data, {3, 3}));
    }
    const Zonotope ZI(I.center(), I.rad().diag()), ZJ(J.center(), J.rad().diag());
    const Interval via_zonotope = ZI.quad_map(Q).interval();
    const Interval R = I.quad_map(Q);
    check(close(R.inf(), via_zonotope.inf(), 1e-12) && close(R.sup(), via_zonotope.sup(), 1e-12),
          b + ": the hull of the zonotope's map");
    const Interval M = I.quad_map(J, Q);
    const Tensor x = ZI.rand_point(50, rng), y = ZJ.rand_point(50, rng);
    for (int k = 0; k < 50; ++k) {
        const Tensor p = x.slice(-1, k, k + 1), q = y.slice(-1, k, k + 1);
        const Tensor image = Tensor::concat({p.matrix_transpose().matmul(Q[0]).matmul(q),
                                             p.matrix_transpose().matmul(Q[1]).matmul(q)},
                                            0);
        check(M.contains(image, 1e-9), b + ": the image of points of both boxes");
    }
}

/// An unbounded box and a matrix of intervals are refused; with the checks off the box is the same.
void checks_and_their_switch() {
    const double inf = std::numeric_limits<double>::infinity();
    const Interval I(column({0.0}), column({1.0}));
    check(test::raises([&] { Interval(column({0.0}), column({inf})).quad_map({Tensor({{1.0}})}); },
                       Error::wrong_value),
          "quad_map: an unbounded box");
    check(
        test::raises([&] { I.quad_map(Interval(column({0.0}), column({inf})), {Tensor({{1.0}})}); },
                     Error::wrong_value),
        "quad_map: an unbounded second box");
    check(test::raises([&] { I.quad_map({Tensor({{1.0, 0.0}, {0.0, 1.0}})}); },
                       Error::dimension_mismatch),
          "quad_map: a matrix of the wrong size");
    ChecksScope off(false);
    check(close(I.quad_map({Tensor({{1.0}})}).sup(), std::vector<double>{1}, 1e-14),
          "quad_map: checks off");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) {
        quad_map_by_hand(b);
        quad_map_is_the_hull_of_the_zonotope_map(b);
    });
    checks_and_their_switch();
    return test::finish("interval quad_map");
}
