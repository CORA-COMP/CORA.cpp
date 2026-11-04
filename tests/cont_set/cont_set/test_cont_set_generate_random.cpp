// test_cont_set_generate_random - ContSet::generate_random: a random set of an admissible class
//
// CORA: contSet/@contSet/generateRandom.m

#include "cora/cont_set/interval/interval.h"
#include "cora/cont_set/set.h"
#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/rng.h"
#include "testing.h"

using namespace cora;
using test::check;

namespace {

/// The class and the dimension follow the arguments; by default both are drawn, and over many
/// draws both classes and dimensions up to 10 occur.
void the_class_and_the_dimension(const std::string &b) {
    Rng rng(3);
    const Set I = ContSet::generate_random(rng, 3, {"interval"});
    check(I.is<Interval>() && I.get().dim() == 3, b + ": a box of dimension 3");
    const Set Z = ContSet::generate_random(rng, 4, {"zonotope"});
    check(Z.is<Zonotope>() && Z.get().dim() == 4 &&
              Z.as<Zonotope>().generators().shape().back() == 8,
          b + ": a zonotope with 2 n generators");
    int boxes = 0, zonotopes = 0, high = 0;
    for (int k = 0; k < 60; ++k) {
        const Set S = ContSet::generate_random(rng);
        boxes += S.is<Interval>();
        zonotopes += S.is<Zonotope>();
        high += S.get().dim() >= 8;
        check(S.get().dim() >= 1 && S.get().dim() <= 10, b + ": the dimension is in 1..10");
    }
    check(boxes > 0 && zonotopes > 0 && boxes + zonotopes == 60 && high > 0,
          b + ": both classes and large dimensions are drawn");
}

/// The same seed gives the same set; an unknown class, no class and dimension 0 are refused; the
/// checks off skip the class check.
void errors_and_their_switch() {
    Rng a(9), c(9);
    const Set S = ContSet::generate_random(a, 3, {"zonotope"}),
              T = ContSet::generate_random(c, 3, {"zonotope"});
    check(test::close(S.get().center(), T.get().center(), 0.0), "the same seed, the same set");
    Rng rng(1);
    check(
        test::raises([&] { ContSet::generate_random(rng, 2, {"polytope"}); }, Error::not_supported),
        "a class that is not implemented");
    check(test::raises([&] { ContSet::generate_random(rng, 2, {}); }, Error::wrong_value),
          "no admissible class");
    check(test::raises([&] { ContSet::generate_random(rng, 0); }, Error::wrong_value),
          "dimension 0");
    ChecksScope off(false);
    check(ContSet::generate_random(rng, 2, {"interval"}).is<Interval>(), "the checks off");
}

} // namespace

int main() {
    test::for_each_backend(the_class_and_the_dimension);
    errors_and_their_switch();
    return test::finish("cont_set generate_random");
}
