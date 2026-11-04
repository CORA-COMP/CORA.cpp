// test_con_zonotope_reduce - ConZonotope::reduce: Girard's rule on the lifted zonotope
//
// CORA: contSet/@conZonotope/reduce.m
// PROPERTY: contSet/@conZonotope/reduce.m
//
// CORA authors:  Dmitry Grebenyuk, Niklas Kochdumper
// Authors:       Claude Sonnet 5.5
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

/// Hand-computed: few enough generators stay; otherwise the long ones stay and a box per row of
/// [G; A] replaces the rest, so that the constraint rows get free factors.
void reduce_known(const std::string &b) {
    const ConZonotope few(column({0.0}), Tensor({{1.0, 2.0}}), Tensor({{1.0, 1.0}}), column({0.5}));
    check(close(few.reduce(2).generators(), {1, 2}, 0), b + ": few generators stay");
    // n = 1, p = 1: more than 1 * 1 + 1 generators are reduced; (order - 1) n = 0 stay
    const ConZonotope many(column({3.0}), Tensor({{1.0, -2.0, 0.5}}), Tensor({{1.0, 1.0, 1.0}}),
                           column({0.5}));
    const ConZonotope r = many.reduce(1);
    check(close(r.generators(), {3.5, 0}, 0), b + ": a box generator for the row of G");
    check(close(r.A(), {0, 3}, 0) && close(r.b(), {0.5}, 0), b + ": and one for the row of A");
    check(close(r.c(), {3}, 0), b + ": the center stays");
    check(close(many.reduce(1, "girard").generators(), r.generators(), 0), b + ": girard is the default");
    // without constraints it is the zonotope's reduction
    const ConZonotope plain(column({0.0, 0.0}), Tensor({{1.0, 0.0, 3.0}, {0.0, 1.0, 1.0}}));
    const Zonotope Z(plain.c(), plain.generators());
    check(close(plain.reduce(1).generators(), Z.reduce(1).generators(), 0),
          b + ": without constraints: the zonotope's reduce");
    check(close(Set(many)->reduce(1).as<ConZonotope>().generators(), r.generators(), 0),
          b + ": through the set interface");
}

/// Stated property: the reduced set holds the sampled points, and has the generators it should.
void reduce_property(const std::string &b) {
    cora::Rng rng(44);
    for (int k = 0; k < 3; ++k) {
        const ConZonotope cz = test::random_con_zonotope(rng, 2, 7, 2);
        const ConZonotope r = cz.reduce(2);
        // order * n + p = 6 < 7 generators: (order - 1) n = 2 stay and n + p = 4 box generators
        check(r.generators().shape() == Shape{2, 6}, b + ": the generator count");
        check(r.contains(test::sample_points(cz, rng, 15)), b + ": the reduced set encloses");
    }
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        reduce_known(b);
        reduce_property(b);
    });
    const ConZonotope a(column({0.0}), Tensor({{1.0}}));
    check(test::raises([&] { a.reduce(0); }, Error::wrong_value), "reduce: order < 1");
    check(test::raises([&] { a.reduce(1, "pca"); }, Error::not_supported), "reduce: other methods");
    ChecksScope off(false);
    check(a.reduce(1).dim() == 1, "reduce: the same with the checks off");
    return test::finish("con_zonotope reduce");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
