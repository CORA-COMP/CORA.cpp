// test_interval_display - interval display: the text that describes an interval
//
// CORA: contSet/@interval/display.m
// CORA authors: Matthias Althoff
// Authors:       Tobias Ladner, Claude Opus 5.5
// Written:       02-November-2026
// Last update:   19-November-2026
// Reviewed:      ---

#include "cora/cont_set/interval/interval.h"
#include "testing.h"

#include <sstream>

using namespace cora;
using test::check;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

void a_box_lists_its_bounds_per_dimension(const std::string &b) {
    const Interval I(test::vec({-1.0, 0.5}), test::vec({1.0, 2.0}));
    const std::string text = I.display();
    check(text.find("interval") == 0 && text.find("- dimension: 2") != std::string::npos,
          b + ": class and dimension");
    check(text.find("[-1, 1]") != std::string::npos && text.find("[0.5, 2]") != std::string::npos,
          b + ": one [inf, sup] row per dimension");
    std::ostringstream out;
    out << I;
    check(out.str() == text, b + ": the stream operator prints the same");
}

void an_interval_matrix_shows_both_bounds(const std::string &b) {
    const Interval M =
        Interval::matrix(Tensor({{0.0, 1.0}, {2.0, 3.0}}), Tensor({{1.0, 2.0}, {3.0, 4.0}}));
    const std::string text = M.display();
    check(text.find("- infimum:") != std::string::npos &&
              text.find("- supremum:") != std::string::npos,
          b + ": both bound matrices");
}

/// display takes no input to check; with the checks off the text is the same.
void checks_and_their_switch() {
    const Interval S = Interval(test::vec({0, -1}), test::vec({1, 1}));
    const std::string on = S.display();
    ChecksScope off(false);
    check(S.display() == on, "display: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        a_box_lists_its_bounds_per_dimension(b);
        an_interval_matrix_shows_both_bounds(b);
    });
    checks_and_their_switch();
    return test::finish("interval display");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
