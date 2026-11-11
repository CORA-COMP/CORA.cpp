// test_level_set_display - LevelSet display: the text of a level set
//
// CORA: contSet/@levelSet/display.m
// CORA authors: Mark Wetzlinger
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
    const LevelSet ls({test::circle(0, 0, 1), Expr::var(0)}, 2, {CompOp::le, CompOp::eq});
    const std::string text = ls.display();
    // the class, the dimension and one comparison per constraint
    check(text.find("levelSet") == 0 && text.find("dimension: 2") != std::string::npos,
          "display names the class and the dimension");
    check(text.find("f <= 0 f == 0") != std::string::npos, "display lists the comparisons");
    ChecksScope off(false);
    check(!ls.display().empty(), "display: checks off");
    return test::finish("level_set display");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
