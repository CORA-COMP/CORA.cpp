// test_stl_get_clauses - the clauses of a formula
//
// CORA: specification/@stl/getClauses.m
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


/// The clauses of a formula.
void run(const std::string &b) {
    const auto x = StlExpr::variables("x", 2);
    const Stl p = x[0] < 5.0, q = x[1] >= 0.0;

    const Stl r = x[0] > 1.0;
    const Stl phi = (p & q) & (r | p);
    const auto cnf = phi.get_clauses();
    check(cnf.size() == 3 && cnf[0].isequal(p) && cnf[1].isequal(q) && cnf[2].isequal(r | p),
          b + ": cnf");
    const auto dnf = (p | q | r).get_clauses("dnf");
    check(dnf.size() == 3 && dnf[2].isequal(r), b + ": dnf");
    check(p.get_clauses().size() == 1 && p.get_clauses("dnf").size() == 1, b + ": one clause");
    // another type is refused
    check(test::raises([&] { p.get_clauses("nnf"); }, Error::wrong_value), b + ": the type");
    {
        ChecksScope off(false);
        check(phi.get_clauses("cnf").size() == 3, b + ": checks off");
    }
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend(run);
    return test::finish("test_stl_get_clauses");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
