// test_zonotope_lift - zonotope lift: the states at given positions of a space of the same size
//
// CORA: contSet/@zonotope/lift_.m
// CORA: contSet/@contSet/lift.m
// PROPERTY: contSet/@zonotope/lift_.m
// CORA authors: Tobias Ladner
// Authors:       Tobias Ladner, Claude Sonnet 5.5, Claude Opus 5.5
// Written:       03-November-2026
// Last update:   04-November-2026
// Reviewed:      ---

#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/rng.h"
#include "testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Hand-computed: state i goes to position dims[i]; the point (1, 2, 3) with dims = [2 3 1] in
/// CORA's numbering becomes (3, 1, 2), where CORA's project gives (2, 3, 1).
void lift_known(const std::string &b) {
    const Zonotope Z(column({1.0, 2.0, 3.0}), Tensor({{1.0, 0.0}, {0.0, 2.0}, {0.0, 0.0}}));
    const Zonotope L = Z.lift(3, {1, 2, 0});
    check(close(L.center(), std::vector<double>{3, 1, 2}, 0),
          b + ": the center, state 0 at position 1");
    check(close(L.generators(), std::vector<double>{0, 0, 1, 0, 0, 2}, 0), b + ": the generators");
    check(close(Z.lift(3, {0, 1, 2}).generators(), Z.generators(), 0), b + ": the identity");
    check(close(Z.lift(3, {1, 0, 2}).center(), std::vector<double>{2, 1, 3}, 0), b + ": a swap");
}

/// Stated property: the support function along d of the lift is that of Z along the pulled-back
/// direction; a batch is member-wise.
void lift_is_a_permutation(const std::string &b) {
    cora::Rng rng(12);
    const Zonotope Z = Zonotope::generate_random(rng, 4, std::nullopt, 5);
    const std::vector<int64_t> dims{2, 0, 3, 1};
    const Zonotope L = Z.lift(4, dims);
    for (int k = 0; k < 20; ++k) {
        const std::vector<double> d = test::random_direction(rng, 4);
        std::vector<double> pulled(4);
        for (int i = 0; i < 4; ++i) pulled[i] = d[dims[i]]; // the direction on the states of Z
        check(test::close(test::support(L, d), test::support(Z, pulled), 1e-12),
              b + ": the support function along the pulled-back direction");
    }
    const Zonotope batch = Zonotope::stack({Z, Z + column({1.0, 1.0, 1.0, 1.0})}).lift(4, dims);
    check(close(batch.center().slice(0, 0, 1).squeeze(0), L.center(), 0),
          b + ": a batch, member 1");
}

/// Wrong dimensions, positions and a smaller space are refused with CORA's identifiers.
void lift_refuses(const std::string &b) {
    const Zonotope Z(column({1.0, 2.0}), Tensor({{1.0}, {0.0}}));
    check(test::raises([&] { Z.lift(1, {0, 0}); }, Error::wrong_value),
          b + ": N below the dimension");
    check(test::raises([&] { Z.lift(3, {0}); }, Error::wrong_value), b + ": one position short");
    check(test::raises([&] { Z.lift(3, {0, 3}); }, Error::wrong_value),
          b + ": a position beyond N");
    check(test::raises([&] { Z.lift(3, {-1, 0}); }, Error::wrong_value),
          b + ": a negative position");
    check(test::raises([&] { Z.lift(3, {0, 2}); }, Error::not_defined),
          b + ": new dimensions would be unbounded");
    check(test::raises([&] { Z.lift(2, {1, 1}); }, Error::wrong_value), b + ": a position twice");
    ChecksScope off(false);
    check(close(Z.lift(2, {1, 0}).center(), std::vector<double>{2, 1}, 0),
          b + ": the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        lift_known(b);
        lift_is_a_permutation(b);
        lift_refuses(b);
    });
    return test::finish("zonotope lift");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
