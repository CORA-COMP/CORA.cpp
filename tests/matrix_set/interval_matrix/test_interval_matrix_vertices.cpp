// test_interval_matrix_vertices - IntervalMatrix vertices: the distinct corner matrices, sorted
//
// CORA: matrixSet/@intervalMatrix/vertices.m
// PROPERTY: matrixSet/@intervalMatrix/vertices.m

#include "interval_matrix_testing.h"

using namespace cora;
using test::check;
using test::close;

namespace {

/// The example has 3 entries with a width, so 8 matrices, sorted by (a11, a21, a12, a22); the
/// point matrix has one vertex, itself.
void vertices_of_the_example(const std::string &b) {
    const std::vector<Tensor> V = test::example().vertices();
    check(V.size() == 1 && V[0].shape() == std::vector<int64_t>{8, 2, 2}, b + ": 8 matrices");
    check(close(V[0], std::vector<double>{1, -1, 0, 2, 1, 1, 0, 2, 1, -1, 3, 2, 1, 1, 3, 2,
                                          2, -1, 0, 2, 2, 1, 0, 2, 2, -1, 3, 2, 2, 1, 3, 2}),
          b + ": the matrices, sorted");
    const Tensor M = test::matrix({1, 2, 3, 4}, 2, 2);
    const std::vector<Tensor> P = IntervalMatrix(M).vertices();
    check(P.size() == 1 && close(P[0].reshape({2, 2}), M), b + ": a point matrix has itself");
}

/// Random 2 x 3 sets: 2^6 distinct matrices, all in the set, with the center as their mean; a
/// batch gives one tensor per member.
void vertices_span_the_set(const std::string &b) {
    Rng rng(18);
    std::vector<double> c(6), d(6);
    rng.normal(c.data(), 6, 0.5);
    rng.uniform(d.data(), 6, 0.1, 1.0);
    const IntervalMatrix A(test::matrix(c, 2, 3), test::matrix(d, 2, 3));
    const Tensor V = A.vertices()[0];
    check(V.shape() == std::vector<int64_t>{64, 2, 3}, b + ": 2^6 vertices");
    bool inside = true;
    for (int64_t k = 0; k < 64; ++k)
        inside = inside && A.contains(V.slice(0, k, k + 1).reshape({2, 3}), "exact", 0.0);
    check(inside, b + ": every vertex is in the set");
    check(close(V.sum(0) * (1.0 / 64), A.center(), 1e-12), b + ": the mean is the center");
    const IntervalMatrix E(Interval(test::matrix({1, -1, 0, 1, 2, 2}, 2, 3),
                                    test::matrix({2, 1, 3, 1, 2, 3}, 2, 3)));
    const std::vector<Tensor> batch =
        IntervalMatrix(Interval::stack({A.interval(), E.interval()})).vertices();
    check(batch.size() == 2 && batch[0].size(0) == 64 && batch[1].size(0) == 16, b + ": a batch");
}

/// Too many entries are refused; with the checks off a valid call gives the same.
void checks_and_their_switch() {
    Rng rng(19);
    check(test::raises([&] { test::random_matrix(rng, 5, 1.0).vertices(); }, Error::not_supported),
          "vertices: 25 entries");
    const Tensor on = test::example().vertices()[0];
    ChecksScope off(false);
    check(close(test::example().vertices()[0], on, 0), "vertices: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend_and_lean([](const std::string &b) {
        vertices_of_the_example(b);
        vertices_span_the_set(b);
    });
    checks_and_their_switch();
    return test::finish("interval_matrix vertices");
}
