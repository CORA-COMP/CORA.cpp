// test_level_set_complement - LevelSet complement: the complement of one inequality
//
// CORA: contSet/@levelSet/not.m
// CORA authors: Niklas Kochdumper
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
    test::for_each_backend([](const std::string &b) {
        // the complement of the closed disc is the open outside: the circle is in the first only
        const LevelSet disc(test::circle(0, 0, 1), 2, CompOp::le);
        const LevelSet outside = disc.complement();
        const Tensor on = Tensor::from_data({1, 0}, {2, 1}),
                     far = Tensor::from_data({2, 0}, {2, 1});
        // the complement excludes what the disc holds on its boundary
        check(disc.contains(on, 0.0) && !outside.contains(on, 0.0),
              b + ": the circle is not outside");
        check(!disc.contains(far, 0.0) && outside.contains(far, 0.0),
              b + ": a far point is outside");
        check(outside.comp_ops().front() == CompOp::lt, b + ": '<=' becomes '<'");
        check(outside.complement().comp_ops().front() == CompOp::le, b + ": and back");
    });
    check(test::raises([] { LevelSet(Expr::var(0), 1, CompOp::eq).complement(); },
                       Error::not_supported),
          "complement: an equality");
    check(test::raises(
              [] {
                  LevelSet({Expr::var(0), Expr::var(0)}, 1, {CompOp::le, CompOp::le}).complement();
              },
              Error::not_supported),
          "complement: two constraints");
    ChecksScope off(false);
    check(LevelSet(Expr::var(0), 1, CompOp::le).complement().dim() == 1, "complement: checks off");
    return test::finish("level_set complement");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
