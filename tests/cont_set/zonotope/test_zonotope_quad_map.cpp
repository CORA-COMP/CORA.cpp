// test_zonotope_quad_map - zonotope quad_map: an enclosure of {(x' Q_i x)_i | x in Z}
//
// CORA: contSet/@zonotope/quadMap.m
// PROPERTY: contSet/@zonotope/quadMap.m

#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/rng.h"
#include "testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

namespace {

/// CORA's unit test: two matrices on a zonotope of two generators, and on two zonotopes.
void quad_map_by_hand(const std::string &b) {
    const Zonotope Z1(column({-4.0, 1.0}), Tensor({{-3.0, -2.0}, {2.0, 3.0}}));
    const Zonotope Z2(column({1.0, -3.0}), Tensor({{4.0, 2.0}, {2.0, -1.0}}));
    const std::vector<Tensor> Q{Tensor({{1.0, -2.0}, {-3.0, 4.0}}), Tensor({{0.5, 0.0}, {2.0, -1.0}})};
    const Zonotope R = Z1.quad_map(Q);
    check(close(R.center(), std::vector<double>{102.5, -16.25}, 1e-13), b + ": the center");
    check(close(R.generators(),
                std::vector<double>{27.5, 35, 95, 110, 125, -5.75, -9.5, -14, -26, -32}, 1e-13),
          b + ": the generators (squares, then the mixed terms)");
    const Zonotope M = Z1.quad_map(Z2, Q);
    check(close(M.center(), std::vector<double>{-43, 3}, 1e-13), b + ": mixed: the center");
    check(close(M.generators(),
                std::vector<double>{-51, -59, -4, -8, -12, -26, -32, -38, 8.5, 14, -2, 6, 14, 1, 7, 13},
                1e-13),
          b + ": mixed: the generators");
}

/// A single nonzero matrix gives one box generator; x^2 on [-1, 1] is enclosed by [0, 2]; a
/// zonotope without generators maps to its point; a batch maps member by member.
void special_cases(const std::string &b) {
    const Zonotope I(column({0.0}), Tensor({{1.0}}));
    const Zonotope R = I.quad_map({Tensor({{2.0}})});
    check(close(R.center(), std::vector<double>{1}, 0) && close(R.generators(), std::vector<double>{1}, 0),
          b + ": 2 x^2 on [-1, 1] is enclosed by [0, 2]");
    const Zonotope Z(column({1.0, 2.0}), Tensor({{1.0}, {0.5}}));
    const Zonotope S = Z.quad_map({Tensor({{1.0, 0.0}, {0.0, 0.0}}), Tensor::zeros({2, 2})});
    check(S.dim() == 2 && S.generators().shape() == std::vector<int64_t>({2, 1}),
          b + ": one nonzero matrix: a box generator");
    check(close(S.generators(), std::vector<double>{2.5, 0}, 1e-14) &&
              close(S.center(), std::vector<double>{1.5, 0}, 1e-14),
          b + ": (1 + b)^2 is enclosed by [-1, 4]; the zero output stays 0");
    const Zonotope P(column({3.0, 1.0}), Tensor::zeros({2, 0}));
    const Zonotope Pq = P.quad_map({Tensor({{1.0, 0.0}, {0.0, 1.0}}), Tensor({{0.0, 1.0}, {0.0, 0.0}})});
    check(close(Pq.center(), std::vector<double>{10, 3}, 0) && Pq.generators().shape().back() == 0,
          b + ": no generators: the image of the point");
    const Zonotope T(column({0.5, 1.0}), Tensor({{0.5, 0.0}, {0.0, 1.0}}));
    const std::vector<Tensor> Q{Tensor({{1.0, 2.0}, {0.0, 1.0}}), Tensor({{0.0, 1.0}, {1.0, 0.0}})};
    const Zonotope batch = Zonotope::stack({Z, T}).quad_map(Q);
    check(close(batch.center().slice(0, 1, 2).squeeze(0), T.quad_map(Q).center(), 1e-14) &&
              close(batch.center().slice(0, 0, 1).squeeze(0), Z.quad_map(Q).center(), 1e-14),
          b + ": a batch");
}

/// Stated property: x' Q_i x of sampled points x of Z (and x1' Q_i x2 of points of two sets) lie
/// in the result.
void the_map_of_the_set_is_enclosed(const std::string &b) {
    cora::Rng rng(61);
    const int n = 3;
    const Zonotope Z = Zonotope::generate_random(rng, n, std::nullopt, 3), Z2 = Zonotope::generate_random(rng, n, std::nullopt, 2);
    std::vector<Tensor> Q;
    for (int i = 0; i < 2; ++i) {
        std::vector<double> data(n * n);
        rng.normal(data.data(), data.size(), 1.0);
        Q.push_back(Tensor::from_data(data, {n, n}));
    }
    const Zonotope R = Z.quad_map(Q), M = Z.quad_map(Z2, Q);
    const Tensor x = Z.rand_point(60, rng, "extreme"), y = Z2.rand_point(60, rng);
    for (int k = 0; k < 60; ++k) {
        const Tensor p = x.slice(-1, k, k + 1), q = y.slice(-1, k, k + 1);
        Tensor image = Tensor::concat({p.matrix_transpose().matmul(Q[0]).matmul(p),
                                       p.matrix_transpose().matmul(Q[1]).matmul(p)}, 0);
        Tensor mixed = Tensor::concat({p.matrix_transpose().matmul(Q[0]).matmul(q),
                                       p.matrix_transpose().matmul(Q[1]).matmul(q)}, 0);
        check(R.contains(image, "exact", 1e-9), b + ": the image of a point of Z");
        check(M.contains(mixed, "exact", 1e-9), b + ": the image of points of Z and Z2");
    }
}

/// Wrong matrices and dimensions are refused; with the checks off the result is the same.
void checks_and_their_switch() {
    const Zonotope Z(column({1.0, 2.0}), Tensor({{1.0}, {0.5}}));
    const Tensor Q = Tensor({{1.0, 0.0}, {0.0, 1.0}});
    check(test::raises([&] { Z.quad_map({}); }, Error::dimension_mismatch), "quad_map: no matrix");
    check(test::raises([&] { Z.quad_map({Tensor({{1.0}})}); }, Error::dimension_mismatch),
          "quad_map: a matrix of the wrong size");
    check(test::raises([&] { Z.quad_map(Zonotope::origin(3), {Q}); }, Error::dimension_mismatch),
          "quad_map: zonotopes of different dimensions");
    const Tensor on = Z.quad_map({Q}).center();
    ChecksScope off(false);
    check(close(Z.quad_map({Q}).center(), on, 0), "quad_map: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) {
        quad_map_by_hand(b);
        special_cases(b);
        the_map_of_the_set_is_enclosed(b);
    });
    checks_and_their_switch();
    return test::finish("zonotope quad_map");
}
