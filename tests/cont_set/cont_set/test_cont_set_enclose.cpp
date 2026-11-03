// test_cont_set_enclose - ContSet::enclose: the checked entry of every set's enclose
//
// CORA: contSet/@contSet/enclose.m

#include "cora/cont_set/empty_set/empty_set.h"
#include "cora/cont_set/interval/interval.h"
#include "cora/cont_set/set.h"
#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/rng.h"
#include "testing.h"
#include "unregistered_set.h"

using namespace cora;
using test::check;
using test::column;

namespace {

/// The hull of two boxes; two zonotopes give one whose support function is at least that of
/// both, and the form enclose(M, Zplus) is the enclosure of the image M Z + Zplus.
void enclose_by_class(const std::string &b) {
    const Interval A(column({0.0, 0.0}), column({1.0, 1.0}));
    const Interval B(column({2.0, -1.0}), column({3.0, 0.0}));
    const Set H = static_cast<const ContSet &>(A).enclose(B);
    check(H.is<Interval>() && test::close(H.as<Interval>().inf(), {0.0, -1.0}) &&
              test::close(H.as<Interval>().sup(), {3.0, 1.0}),
          b + ": the hull of two boxes");

    cora::Rng rng(5);
    const Zonotope Z1 = Zonotope::generate_random(rng, 3, std::nullopt, 4), Z2 = Zonotope::generate_random(rng, 3, std::nullopt, 2);
    const Set E = static_cast<const ContSet &>(Z1).enclose(Z2);
    check(E.is<Zonotope>() && E.as<Zonotope>().isequal(Z1.enclose(Z2)), b + ": a zonotope");
    for (int k = 0; k < 20; ++k) {
        const std::vector<double> d = test::random_direction(rng, 3);
        check(test::support(E, d) >= test::support(Z1, d) - 1e-9 &&
                  test::support(E, d) >= test::support(Z2, d) - 1e-9,
              b + ": the support function of both is below");
    }

    const Tensor M = Tensor::from_data({1.0, 2.0, 0.0, 0.5, -1.0, 1.0, 0.0, 0.0, 2.0}, {3, 3});
    const Zonotope Zplus = Zonotope::generate_random(rng, 3, std::nullopt, 2);
    check(Z1.enclose(M, Zplus).isequal(Z1.enclose(Z1.linear_map(M) + Zplus)),
          b + ": enclose(M, Zplus)");
}

/// A dimension mismatch, a pair of classes without it, the checks off.
void errors_and_their_switch() {
    const Interval A(column({0.0, 0.0}), column({1.0, 1.0}));
    const Zonotope Z(column({0.0, 0.0}), Tensor::from_data({1.0, 0.0, 0.0, 1.0}, {2, 2}));
    const Zonotope Z3(column({0.0, 0.0, 0.0}), Tensor::from_data({1.0, 0.0, 0.0}, {3, 1}));
    check(test::raises([&] { static_cast<const ContSet &>(A).enclose(Z3); },
                       Error::dimension_mismatch),
          "different dimensions");
    check(test::raises([&] { static_cast<const ContSet &>(A).enclose(Z); }, Error::noops),
          "a box and a zonotope");
    check(test::raises([&] { static_cast<const ContSet &>(Z).enclose(A); }, Error::noops),
          "a zonotope and a box");
    check(test::raises([&] { static_cast<const ContSet &>(EmptySet(2)).enclose(A); }, Error::noops),
          "the empty set");
    check(test::raises([&] { test::UnregisteredSet().enclose(A); }, Error::noops), "noops");
    check(test::raises([&] { Z.enclose(Tensor::from_data({1.0, 0.0}, {1, 2}), Z); },
                       Error::dimension_mismatch),
          "M not square");
    ChecksScope off(false);
    check(static_cast<const ContSet &>(A).enclose(A).is<Interval>(), "the checks off");
}

} // namespace

int main() {
    test::for_each_backend(enclose_by_class);
    errors_and_their_switch();
    return test::finish("cont_set enclose");
}
