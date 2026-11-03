// test_zonotope_hausdorff_distance_to_box - zonotope hausdorff_distance_to_box: distance to the box
//
// CORA: contSet/@zonotope/dH2box.m
// PROPERTY: contSet/@zonotope/dH2box.m

#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/rng.h"
#include "testing.h"

#include <cmath>

using namespace cora;
using test::check;
using test::close;
using test::column;

namespace {

/// One generator (1, 1): the box has radius (1, 1), every method has a hand-computed value.
void each_method_on_one_generator(const std::string &b) {
    const Zonotope Z(column({0.0, 0.0}), Tensor({{1.0}, {1.0}}));
    check(close(Z.hausdorff_distance_to_box("naive").item(), std::sqrt(2.0), 1e-14), b + ": naive");
    check(close(Z.hausdorff_distance_to_box("wgreedy").item(), 2.0, 1e-14), b + ": wgreedy");
    check(close(Z.hausdorff_distance_to_box("wopt").item(), std::sqrt(2.0), 1e-14), b + ": wopt");
    check(close(Z.hausdorff_distance_to_box("exact").item(), std::sqrt(2.0), 1e-4), b + ": exact");
    check(
        close(Z.hausdorff_distance_to_box().item(), Z.hausdorff_distance_to_box("exact").item(), 0),
        b + ": the default of a plane is exact");
    const Zonotope Z4(column({0.0, 0.0, 0.0, 0.0}), Tensor({{1.0}, {1.0}, {1.0}, {1.0}}));
    check(close(Z4.hausdorff_distance_to_box().item(), 2.0, 1e-14),
          b + ": the default above 3-D is naive");
    const Zonotope Z1(column({0.0}), Tensor({{1.0, -2.0}}));
    check(close(Z1.hausdorff_distance_to_box("exact").item(), 0.0, 0),
          b + ": a box in one dimension");
}

/// A batch gives one number per member.
void a_batch_gives_a_number_per_member(const std::string &b) {
    const Zonotope Z(column({0.0, 0.0}), Tensor({{1.0}, {1.0}}));
    const Zonotope Z2(column({0.0, 0.0}), Tensor({{2.0}, {0.0}}));
    const Tensor d = Zonotope::stack({Z, Z2}).hausdorff_distance_to_box("wopt");
    check(d.shape() == std::vector<int64_t>({2, 1, 1}), b + ": shape (2, 1, 1)");
    check(close(d.data()[1], 0.0, 1e-14), b + ": a segment along an axis is its box");
}

/// Stated property: the distance is at most the sum of the lengths of the generators (the bound
/// of [Althoff 2010]), and zero for a box.
void bounded_by_the_generator_lengths(const std::string &b) {
    cora::Rng rng(24);
    for (int n = 2; n <= 3; ++n) {
        const Zonotope Z = Zonotope::generate_random(rng, n, std::nullopt, 5);
        double sum = 0;
        const std::vector<double> G = Z.generators().data();
        for (int j = 0; j < 5; ++j) {
            double len = 0;
            for (int i = 0; i < n; ++i) len += G[i * 5 + j] * G[i * 5 + j];
            sum += std::sqrt(len);
        }
        check(Z.hausdorff_distance_to_box("exact").item() <= sum + 1e-9,
              b + ": exact <= sum of lengths");
        check(Z.hausdorff_distance_to_box("naive").item() <= sum + 1e-9,
              b + ": naive <= sum of lengths");
        check(Z.hausdorff_distance_to_box("exact").item() >= 0, b + ": nonnegative");
    }
    const Zonotope box(column({0.0, 0.0}), Tensor({{1.0, 0.0}, {0.0, 2.0}}));
    check(close(box.hausdorff_distance_to_box("exact").item(), 0.0, 1e-12),
          b + ": a box has distance 0");
}

/// Unknown methods and "ell" are refused; with the checks off the result is the same.
void checks_and_their_switch() {
    const Zonotope Z(column({0.0, 0.0}), Tensor({{1.0}, {1.0}}));
    check(test::raises([&] { Z.hausdorff_distance_to_box("fast"); }, Error::wrong_value),
          "hausdorff_distance_to_box: unknown method");
    check(test::raises([&] { Z.hausdorff_distance_to_box("ell"); }, Error::not_supported),
          "hausdorff_distance_to_box: ell");
    const double on = Z.hausdorff_distance_to_box("wopt").item();
    ChecksScope off(false);
    check(Z.hausdorff_distance_to_box("wopt").item() == on,
          "hausdorff_distance_to_box: checks off");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) {
        each_method_on_one_generator(b);
        a_batch_gives_a_number_per_member(b);
        bounded_by_the_generator_lengths(b);
    });
    checks_and_their_switch();
    return test::finish("zonotope hausdorff_distance_to_box");
}
