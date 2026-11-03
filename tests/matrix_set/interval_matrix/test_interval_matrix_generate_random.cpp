// test_interval_matrix_generate_random - IntervalMatrix generate_random: random interval matrices
//
// CORA: matrixSet/@intervalMatrix/generateRandom.m
// PROPERTY: matrixSet/@intervalMatrix/generateRandom.m

#include "interval_matrix_testing.h"

using namespace cora;
using test::check;
using test::close;

namespace {

/// Without options the size is random in 1..10, the radii are in [0, 10], the same seed gives the
/// same set.
void random_sets_have_the_stated_ranges(const std::string &b) {
    Rng rng(28), again(28);
    bool sizes = true, radii = true;
    for (int trial = 0; trial < 30; ++trial) {
        const IntervalMatrix A = IntervalMatrix::generate_random(rng);
        sizes = sizes && A.rows() >= 1 && A.rows() <= 10 && A.cols() >= 1 && A.cols() <= 10;
        radii = radii && A.rad().min().item() >= 0 && A.rad().max().item() <= 10 + 1e-9;
    }
    check(sizes, b + ": sizes in 1..10");
    check(radii, b + ": radii in [0, 10]");
    Rng a(5), c(5);
    const IntervalMatrix X = IntervalMatrix::generate_random(a),
                         Y = IntervalMatrix::generate_random(c);
    check(close(X.interval().inf(), Y.interval().inf(), 0) &&
              close(X.interval().sup(), Y.interval().sup(), 0),
          b + ": the same seed");
}

/// dimension, center and max_radius: a given dimension, the shape of the center, a number as the
/// radius of every entry.
void options_are_used(const std::string &b) {
    Rng rng(29);
    const IntervalMatrix D = IntervalMatrix::generate_random(rng, std::array<int64_t, 2>{3, 4});
    check(D.dim() == std::array<int64_t, 2>{3, 4}, b + ": dimension");
    const Tensor center = test::matrix({1, 2, 3, 4, 5, 6}, 2, 3);
    const Tensor half = test::matrix({0.5}, 1, 1);
    const IntervalMatrix C = IntervalMatrix::generate_random(rng, std::nullopt, center, half);
    check(C.dim() == std::array<int64_t, 2>{2, 3} && close(C.center(), center, 1e-12) &&
              close(C.rad(), std::vector<double>(6, 0.5), 1e-12),
          b + ": a center and one radius");
    const Tensor radius = test::matrix({0, 1, 2, 3}, 2, 2);
    const IntervalMatrix R =
        IntervalMatrix::generate_random(rng, std::nullopt, std::nullopt, radius);
    check(R.dim() == std::array<int64_t, 2>{2, 2} && close(R.rad(), radius, 1e-12),
          b + ": radii give the shape");
}

/// A negative radius, a dimension below 1 and a radius of another shape than the center are
/// refused; with the checks off a valid call gives the same.
void checks_and_their_switch() {
    Rng rng(30);
    const Tensor negative = test::matrix({-1}, 1, 1);
    check(test::raises(
              [&] { IntervalMatrix::generate_random(rng, std::nullopt, std::nullopt, negative); },
              Error::wrong_value),
          "generate_random: a negative radius");
    check(test::raises([&] { IntervalMatrix::generate_random(rng, std::array<int64_t, 2>{0, 2}); },
                       Error::wrong_value),
          "generate_random: dimension 0");
    check(test::raises(
              [&] {
                  IntervalMatrix::generate_random(rng, std::nullopt, test::matrix({1, 2}, 1, 2),
                                                  test::matrix({1, 2, 3, 4}, 2, 2));
              },
              Error::wrong_input_in_constructor),
          "generate_random: a center and a radius of two shapes");
    Rng a(3), c(3);
    const IntervalMatrix on = IntervalMatrix::generate_random(a);
    ChecksScope off(false);
    check(close(IntervalMatrix::generate_random(c).interval().inf(), on.interval().inf(), 0),
          "generate_random: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend_and_lean([](const std::string &b) {
        random_sets_have_the_stated_ranges(b);
        options_are_used(b);
    });
    checks_and_their_switch();
    return test::finish("interval_matrix generate_random");
}
