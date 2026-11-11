// test_level_set_noops - LevelSet: the operations a level set does not have
//
// CORA.cpp only: CORA's contSet raises noops for each of them
// Authors:       Claude Sonnet 5.5
// Written:       11-November-2026
// Last update:   ---
// Reviewed:      ---

#include "level_set_testing.h"

using namespace cora;
using test::check;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //


// ===========================================  MAIN  =========================================== //

int main() {
    const LevelSet ls(test::circle(0, 0, 1), 2, CompOp::eq);
    Rng rng(1);
    check(test::raises([&] { ls.center(); }, Error::noops), "center");
    check(test::raises([&] { ls.interval(); }, Error::noops), "interval");
    check(test::raises([&] { ls.support_function(test::column({1.0, 0.0})); }, Error::noops),
          "support_function");
    check(test::raises([&] { ls.rand_point(1, rng); }, Error::noops), "rand_point");
    check(test::raises([&] { ls.project({0}); }, Error::noops), "project");
    check(test::raises([&] { ls.vertices(); }, Error::noops), "vertices");
    const LevelSet ls3(Expr::var(0) + Expr::var(1) + Expr::var(2), 3, CompOp::le);
    check(test::raises([&] { ls3.vertices_3d(); }, Error::noops), "vertices_3d");
    ChecksScope off(false);
    check(test::raises([&] { ls.center(); }, Error::noops), "center, checks off");
    return test::finish("level_set noops");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
