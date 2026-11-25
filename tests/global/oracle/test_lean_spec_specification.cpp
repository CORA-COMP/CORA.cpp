// test_lean_spec_specification - project and isequal of Specification against the CORALean oracle
//
// The specification's own steps are bookkeeping (the type, the window, the locations); its numbers
// come from the set. project of a box and of a zonotope specification is bit for bit the oracle's
// interval.project and zonotope.project; isequal of two box specifications is the oracle's
// interval.isequal of the sets where the types and windows agree, and of two logic specifications
// the oracle's stl.isequal (eq and ne are isequal).
//
// ORACLE: specification/@specification/eq.m
// ORACLE: specification/@specification/isequal.m
// ORACLE: specification/@specification/ne.m
// ORACLE: specification/@specification/project.m
// Authors:       Claude Opus 5.5
// Written:       25-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/interval/interval.h"
#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/oracle/interval.h"
#include "cora/global/oracle/stl.h"
#include "cora/global/oracle/zonotope.h"
#include "cora/specification/specification.h"
#include "oracle_testing.h"

using namespace cora;
namespace oracle = test::oracle;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //


// ===========================================  MAIN  =========================================== //

int main() {
    if (!oracle::available()) return 0;
    // the set of a projected specification is the projected set
    oracle::run("interval.project", [](oracle::Random &rnd) {
        for (const std::vector<int64_t> &dims :
             {std::vector<int64_t>{0}, std::vector<int64_t>{2, 0}, std::vector<int64_t>{1, 2}}) {
            const Interval I = rnd.box(3);
            const Zonotope Z = rnd.zonotope(3, 4);
            const Specification boxes = Specification::safe_set(I).project(dims);
            const Specification zonos = Specification::unsafe_set(Z).with_time(0, 1).project(dims);
            oracle::check("project (box)", boxes.set().as<Interval>(),
                          lean::Interval(I).project(dims));
            oracle::check("project (zonotope)", zonos.set().as<Zonotope>(),
                          lean::Zonotope(Z).project(dims));
            test::check(zonos.type() == SpecType::unsafe_set && zonos.time().has_value(),
                        "project: the type and the window stay");
        }
    });
    // isequal: the same box, a moved one, another type, another window
    oracle::run("interval.isequal", [](oracle::Random &rnd) {
        for (int trial = 0; trial < 12; ++trial) {
            const Interval I = rnd.box(2);
            const Interval J = trial % 2 == 0 ? I : rnd.box(2);
            const Specification a = Specification::safe_set(I);
            const bool sets = lean::Interval(I).isequal(lean::Interval(J), 0);
            oracle::check("isequal", a.isequal(Specification::safe_set(J), 0.0), sets);
            oracle::check("isequal (type)", a.isequal(Specification::unsafe_set(J), 0.0), false);
            oracle::check(
                "isequal (window)",
                a.with_time(0, 1).isequal(Specification::safe_set(J).with_time(0, 1), 0.0), sets);
        }
    });
    // two logic specifications: their formulas
    oracle::run("stl.isequal", [](oracle::Random &rnd) {
        const std::vector<StlExpr> x = StlExpr::variables("x", 2);
        for (int trial = 0; trial < 12; ++trial) {
            const Stl a = (x[0] < rnd.scalar()).finally(StlInterval(0, 1));
            const Stl b = (x[1] >= rnd.scalar()).or_(Stl(trial % 3 == 0));
            const Stl phi = a.and_(b), psi = trial % 2 == 0 ? b.and_(a) : a.or_(b);
            const bool theirs =
                lean::ask_stl("stl.isequal", phi, {{"psi", lean::Wire(lean::stl_json(psi), "")}})
                    .get<bool>("value");
            oracle::check("isequal (logic)",
                          Specification::logic(phi).isequal(Specification::logic(psi)), theirs);
        }
    });
    return test::finish("lean spec specification");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
