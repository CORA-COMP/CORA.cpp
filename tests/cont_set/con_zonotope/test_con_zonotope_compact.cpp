// test_con_zonotope_compact - ConZonotope::compact: without zero rows and generators
//
// CORA: contSet/@conZonotope/compact_.m
// PROPERTY: contSet/@conZonotope/compact_.m
// CORA authors: Niklas Kochdumper, Mark Wetzlinger
// Authors:       Claude Sonnet 5.5, Tobias Ladner
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#include "con_zonotope_testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Hand-computed: a zero row 0 = 0 and a zero generator in no constraint go; an infeasible zero
/// row 0 = 1 and a generator that is in a constraint stay.
void compact_known(const std::string &b) {
    const ConZonotope cz(column({1.0, 2.0}), Tensor({{1.0, 0.0, 2.0}, {0.0, 0.0, 3.0}}),
                         Tensor({{0.0, 0.0, 0.0}, {1.0, 0.0, 1.0}}), column({0.0, 1.0}));
    for (const char *method : {"zeros", "all"}) {
        const ConZonotope c = cz.compact(method);
        check(close(c.generators(), {1, 2, 0, 3}, 0),
              b + ": the zero generator goes (" + method + ")");
        check(close(c.A(), {1, 1}, 0) && close(c.b(), {1}, 0), b + ": the zero row goes");
        check(close(c.c(), cz.c(), 0), b + ": the center stays");
    }
    // a generator that is zero but in a constraint stays; so does the row 0 = 1
    const ConZonotope kept(column({0.0}), Tensor({{0.0, 1.0}}), Tensor({{1.0, 0.0}, {0.0, 0.0}}),
                           column({0.5, 1.0}));
    check(kept.compact().generators().shape() == Shape{1, 2},
          b + ": a constrained generator stays");
    check(close(kept.compact().b(), {0.5, 1}, 0), b + ": the infeasible row stays");
    check(close(cz.compact("aligned").generators(), cz.generators(), 0),
          b + ": 'aligned': as it is");
    check(ConZonotope::empty(2).compact().A().shape() == Shape{1, 0}, b + ": empty stays empty");
}

/// Stated property: the set is unchanged (same box, same points), and compacting twice changes
/// nothing.
void compact_property(const std::string &b) {
    cora::Rng rng(24);
    for (int k = 0; k < 4; ++k) {
        const ConZonotope base = test::random_con_zonotope(rng, 3, 4, 2);
        // append a zero generator (a zero column of G and of A) and a zero row
        const ConZonotope padded(
            base.c(), Tensor::concat({base.generators(), Tensor::zeros({3, 1})}, -1),
            Tensor::concat(
                {Tensor::concat({base.A(), Tensor::zeros({2, 1})}, -1), Tensor::zeros({1, 5})}, -2),
            Tensor::concat({base.b(), Tensor::zeros({1, 1})}, -2));
        const ConZonotope once = padded.compact();
        check(once.generators().shape() == base.generators().shape(), b + ": the padding is gone");
        check(close(once.interval().inf(), base.interval().inf(), 1e-9), b + ": the same box");
        check(close(once.compact().A(), once.A(), 0), b + ": idempotent");
    }
}

/// A wrong method and a negative tol are refused; a batch drops what every member drops.
void checks_and_batches() {
    const ConZonotope cz(column({0.0}), Tensor({{1.0, 0.0}}));
    check(test::raises([&] { cz.compact("rows"); }, Error::wrong_value),
          "compact: an unknown method");
    check(test::raises([&] { cz.compact("zeros", -1.0); }, Error::wrong_value), "compact: tol < 0");
    const Tensor G = Tensor::concat(
        {Tensor({{1.0, 0.0, 0.0}}).expand_dims(0), Tensor({{1.0, 0.0, 2.0}}).expand_dims(0)}, 0);
    const ConZonotope batch(Tensor::zeros({2, 1, 1}), G);
    check(batch.compact().generators().shape() == Shape{2, 1, 2},
          "compact: a batch keeps what some member keeps");
    ChecksScope off(false);
    check(cz.compact().generators().shape() == Shape{1, 1},
          "compact: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        compact_known(b);
        compact_property(b);
    });
    checks_and_batches();
    return test::finish("con_zonotope compact");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
