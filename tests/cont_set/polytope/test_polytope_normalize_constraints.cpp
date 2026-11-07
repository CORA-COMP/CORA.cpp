// test_polytope_normalize_constraints - Polytope::normalize_constraints: rows to unit scale
//
// CORA: contSet/@polytope/normalizeConstraints.m
// CORA: contSet/@polytope/private/priv_normalizeConstraints.m
// PROPERTY: contSet/@polytope/normalizeConstraints.m
//
// Property: the set is unchanged, and the offsets are +-1 (or the rows unit)
//
// CORA authors: Viktor Kotsev, Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "polytope_testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The two normalizations on hand-computed rows.
void scalings(const std::string &b) {
    const Polytope P(Tensor({{2.0, 0.0}, {0.0, -4.0}, {3.0, 4.0}}), column({4.0, -2.0, 0.0}),
                     Tensor({{0.0, 2.0}}), column({-6.0}));
    const Polytope by_offset = P.normalize_constraints();
    check(close(by_offset.A(), {0.5, 0.0, 0.0, -2.0, 3.0, 4.0}) &&
              close(by_offset.b(), {1.0, -1.0, 0.0}),
          b + ": b: the offsets to +-1, zero stays");
    check(close(by_offset.Ae(), {0.0, -1.0 / 3.0}) && close(by_offset.be(), {1.0}),
          b + ": b: an equality to 1");
    const Polytope by_length = P.normalize_constraints("A");
    check(close(by_length.A(), {1.0, 0.0, 0.0, -1.0, 0.6, 0.8}) &&
              close(by_length.b(), {2.0, -0.5, 0.0}),
          b + ": A: unit rows");
    check(close(by_length.Ae(), {0.0, 1.0}) && close(by_length.be(), {-3.0}),
          b + ": A: the equality");
    check(close(P.normalize_constraints("be").A(), by_offset.A(), 0) &&
              close(P.normalize_constraints("Ae").A(), by_length.A(), 0),
          b + ": be and Ae are b and A");
}

/// The normalized set is the same set.
void properties(const std::string &b) {
    Rng rng(4);
    bool same = true;
    const Polytope P = test::random_polytope(rng, 3, 5);
    for (const char *type : {"b", "A"}) {
        const Polytope N = P.normalize_constraints(type);
        for (int k = 0; k < 30; ++k) {
            const Tensor x = column(test::random_direction(rng, 3));
            same = same && N.contains(x) == P.contains(x);
        }
    }
    check(same, b + ": the same points");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        scalings(b);
        properties(b);
    });
    check(test::raises([] { test::square_h().normalize_constraints("c"); }, Error::wrong_value),
          "normalize_constraints: the type");
    check(test::raises([] { test::square_v().normalize_constraints(); }, Error::special_error),
          "normalize_constraints: a vertex polytope has no rows");
    ChecksScope off(false);
    check(test::square_h().normalize_constraints().b().max().item() == 1.0,
          "normalize_constraints: the same with the checks off");
    return test::finish("polytope normalize_constraints");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
