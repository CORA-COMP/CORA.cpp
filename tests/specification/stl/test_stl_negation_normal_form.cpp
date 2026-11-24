// test_stl_negation_normal_form - the negation normal form of a formula
//
// CORA: specification/@stl/negationNormalForm.m
// PROPERTY: specification/@stl/negationNormalForm.m
// CORA authors: Florian Lercher
// Authors:       Claude Sonnet 5.5
// Written:       24-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/global/settings.h"
#include "cora/specification/stl.h"
#include "stl_testing.h"
#include "testing.h"

using namespace cora;
using test::check;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {


/// The negation normal form of a formula.
void run(const std::string &b) {
    const auto x = StlExpr::variables("x", 2);
    const Stl p = x[0] < 5.0, q = x[1] >= 0.0;

    const StlInterval w = StlInterval(0, 2);
    // a negated comparison is the comparison that fails (CORA keeps the strictness)
    check((~p).negation_normal_form().isequal(x[0] >= 5.0), b + ": lt");
    check((~(x[0] <= 5.0)).negation_normal_form().isequal(x[0] > 5.0), b + ": le");
    check((~q).negation_normal_form().isequal(x[1] < 0.0), b + ": ge");
    check((~(x[0] > 5.0)).negation_normal_form().isequal(x[0] <= 5.0), b + ": gt");
    // De Morgan and the duals of the temporal operators
    check((~(p & q)).negation_normal_form().isequal((x[0] >= 5.0) | (x[1] < 0.0)),
          b + ": De Morgan");
    check((~p.finally(w)).negation_normal_form().isequal((x[0] >= 5.0).globally(w)),
          b + ": finally");
    check((~p.globally(w)).negation_normal_form().isequal((x[0] >= 5.0).finally(w)),
          b + ": globally");
    check((~p.until(q, w)).negation_normal_form().isequal((x[0] >= 5.0).release(x[1] < 0.0, w)),
          b + ": until");
    check((~p.release(q, w)).negation_normal_form().isequal((x[0] >= 5.0).until(x[1] < 0.0, w)),
          b + ": release");
    check((~p.next(1)).negation_normal_form().isequal((x[0] >= 5.0).next(1)), b + ": next");
    check((~Stl(true)).negation_normal_form().isequal(Stl(false)) &&
              (~~p).negation_normal_form().isequal(p),
          b + ": constants and double negation");
    // random formulas keep their meaning, and ~ only stays below no operator
    Rng rng(4);
    for (int k = 0; k < 40; ++k) {
        const Stl phi = stl_test::random_formula(rng, 4);
        check(stl_test::equivalent(phi, phi.negation_normal_form()), b + ": equivalent");
    }
    {
        ChecksScope off(false);
        check((~p).negation_normal_form().isequal(x[0] >= 5.0), b + ": checks off");
    }
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend(run);
    return test::finish("test_stl_negation_normal_form");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
