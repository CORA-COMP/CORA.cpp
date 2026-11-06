// test_zonotope_interval - zonotope interval: the smallest box around a zonotope
//
// CORA: contSet/@zonotope/interval.m
// CORA authors: Matthias Althoff
// Authors:       Tobias Ladner
// Written:       02-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/rng.h"
#include "testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;
using test::random_direction;
using test::support;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

void hull_of_a_known_zonotope(const std::string &b) {
    const Zonotope Z(column({1.0, 2.0}), Tensor({{1.0, 0.0, 0.5}, {0.0, 2.0, 0.5}}));
    check(close(Z.interval().inf(), std::vector<double>{-0.5, -0.5}),
          b + ": the hull's lower corner");
    check(close(Z.interval().sup(), std::vector<double>{2.5, 4.5}),
          b + ": the hull's upper corner");
}

void interval_is_the_hull(const std::string &b) {
    cora::Rng rng(6);
    const Zonotope Z = Zonotope::generate_random(rng, 4, std::nullopt, 6);
    const Interval I = Z.interval();
    for (int i = 0; i < 4; ++i) {
        std::vector<double> e(4, 0.0), f(4, 0.0);
        e[i] = 1.0;
        f[i] = -1.0;
        check(close(I.support_function(column(e)).data()[0], support(Z, e), 1e-12),
              b + ": hull, upper");
        check(close(I.support_function(column(f)).data()[0], support(Z, f), 1e-12),
              b + ": hull, lower");
    }
    for (int k = 0; k < 10; ++k) {
        const std::vector<double> d = random_direction(rng, 4);
        check(I.support_function(column(d)).data()[0] >= support(Z, d) - 1e-10,
              b + ": the hull contains Z");
    }
}

/// interval takes no input to check; with the checks off the box is the same.
void checks_and_their_switch() {
    const Zonotope Z = Zonotope(Tensor({1.0, 0.0}), Tensor({{0.5, 0.0}, {0.0, 0.5}}));
    const Tensor on = Z.interval().sup();
    ChecksScope off(false);
    check(test::close(Z.interval().sup(), on, 0), "interval: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        hull_of_a_known_zonotope(b);
        interval_is_the_hull(b);
    });
    checks_and_their_switch();
    return test::finish("zonotope interval");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
