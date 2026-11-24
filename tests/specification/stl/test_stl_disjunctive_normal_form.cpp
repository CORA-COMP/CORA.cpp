// test_stl_disjunctive_normal_form - the disjunctive normal form of a formula
//
// CORA: specification/@stl/disjunctiveNormalForm.m
// PROPERTY: specification/@stl/disjunctiveNormalForm.m
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

/// Whether a | node (a & node, if inside) occurs below a & node.
bool below(const Stl &phi, bool inside) {
    if (phi.kind() == StlKind::not_) return below(phi.lhs(), inside);
    if (phi.kind() != StlKind::and_ && phi.kind() != StlKind::or_) return false;
    if (phi.kind() == StlKind::or_ && inside) return true;
    const bool in = inside || phi.kind() == StlKind::and_;
    return below(phi.lhs(), in) || below(phi.rhs(), in);
}


/// The disjunctive normal form of a formula.
void run(const std::string &b) {
    const auto x = StlExpr::variables("x", 2);
    const Stl p = x[0] < 5.0, q = x[1] >= 0.0;

    const Stl r = x[0] > 1.0;
    // distribute & over |: (p | q) & r
    const Stl res = (p | q) & r;
    const Stl nf = res.disjunctive_normal_form();
    check(nf.isequal((r & p) | (r & q)) || nf.isequal((p & r) | (q & r)), b + ": one operand");
    check(stl_test::equivalent(res, nf), b + ": same meaning");
    check(p.disjunctive_normal_form().isequal(p) &&
              (p & q).disjunctive_normal_form().isequal(p & q),
          b + ": already in form");
    // the flag says the formula is in negation normal form already
    check((p | q).disjunctive_normal_form(StlNnf::given).isequal(p | q), b + ": given");
    // random formulas keep their meaning and no | stays below a &
    Rng rng(4);
    for (int k = 0; k < 40; ++k) {
        const Stl phi = stl_test::random_formula(rng, 3);
        const Stl nf2 = phi.disjunctive_normal_form();
        check(stl_test::equivalent(phi, nf2), b + ": equivalent");
        check(!below(nf2, false), b + ": the form");
    }
    {
        ChecksScope off(false);
        check(res.disjunctive_normal_form().kind() == StlKind::or_, b + ": checks off");
    }
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend(run);
    return test::finish("test_stl_disjunctive_normal_form");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
