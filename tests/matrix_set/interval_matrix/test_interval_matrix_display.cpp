// test_interval_matrix_display - IntervalMatrix display: the text of a set
//
// CORA: matrixSet/@intervalMatrix/display.m
// PROPERTY: matrixSet/@intervalMatrix/display.m

#include "interval_matrix_testing.h"

#include <sstream>

using namespace cora;
using test::check;

namespace {

/// The text names the class and the dimension, then both bound matrices; << prints it.
void text_of_the_example(const std::string &b) {
    const std::string text = test::example().display();
    check(text.rfind("intervalMatrix\n- dimension: 2  2\n- infimum:\n", 0) == 0, b + ": the head");
    check(text.find("- supremum:\n") != std::string::npos, b + ": the supremum");
    std::ostringstream out;
    out << test::example();
    check(out.str() == text, b + ": operator<< prints display()");
}

/// The text shows the numbers of the bounds, and a larger matrix shows more of them.
void text_shows_the_bounds(const std::string &b) {
    Rng rng(26);
    const std::string small = test::example().display();
    check(small.find("-1") != std::string::npos && small.find('3') != std::string::npos, b + ": bounds");
    const std::string large = test::random_matrix(rng, 4, 1.0).display();
    check(large.size() > small.size() && large.find("dimension: 4  4") != std::string::npos,
          b + ": a 4 x 4 set");
}

/// display has no input to refuse; with the checks off the text is the same.
void checks_and_their_switch() {
    const std::string on = test::example().display();
    ChecksScope off(false);
    check(test::example().display() == on, "display: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend_and_lean([](const std::string &b) {
        text_of_the_example(b);
        text_shows_the_bounds(b);
    });
    checks_and_their_switch();
    return test::finish("interval_matrix display");
}
