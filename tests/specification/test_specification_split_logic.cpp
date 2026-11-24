// test_specification_split_logic - the logic specifications of a list apart from the others
//
// CORA: specification/@specification/splitLogic.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Sonnet 5.5
// Written:       24-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/polytope/polytope.h"
#include "cora/global/settings.h"
#include "cora/specification/specification.h"
#include "testing.h"

using namespace cora;
using test::check;
using test::vec;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The logic specifications of a list apart.
void run(const std::string &b) {
    const auto x = StlExpr::variables("x", 2);
    const Stl phi = (x[0] < 5.0).finally(StlInterval(0, 2));
    const Polytope P(Tensor({{1.0, 0.0}}), vec({1.0}));
    const Specification safe = Specification::safe_set(P), unsafe = Specification::unsafe_set(P);
    const Specification logic = Specification::logic(phi);

    const auto [non_logic, only_logic] = split_logic({safe, logic, unsafe, logic});
    check(non_logic.size() == 2 && non_logic[0].type() == SpecType::safe_set &&
              non_logic[1].type() == SpecType::unsafe_set,
          b + ": sets");
    check(only_logic.size() == 2 && only_logic[0].formula().isequal(phi), b + ": logic");
    check(split_logic({}).first.empty() && split_logic({}).second.empty(), b + ": empty");
    check(logic.isequal(Specification::logic(phi)) && !logic.isequal(safe), b + ": isequal");

    // a logic specification has no set and no check on sets
    check(test::raises([&] { logic.set(); }, Error::wrong_value), b + ": no set");
    check(test::raises([&] { safe.formula(); }, Error::wrong_value), b + ": no formula");
    check(test::raises([&] { logic.check_points(Tensor({{0.0, 0.0}})); }, Error::not_supported),
          b + ": points");
    check(test::raises([&] { logic.project({0}); }, Error::not_supported), b + ": project");
    {
        ChecksScope off(false);
        check(split_logic({logic}).second.size() == 1, b + ": checks off");
    }
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend(run);
    return test::finish("test_specification_split_logic");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
