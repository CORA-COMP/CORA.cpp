// test_empty_set_display - EmptySet display: the text that describes the empty set
//
// CORA: contSet/@emptySet/display.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/empty_set/empty_set.h"
#include "testing.h"

#include <sstream>

using namespace cora;
using test::check;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

void the_text_names_the_dimension(const std::string &b) {
    const std::string text = EmptySet(3).display();
    check(text.find("emptySet") == 0 && text.find("3-dimensional empty set") != std::string::npos,
          b + ": class and dimension");
    std::ostringstream out;
    out << EmptySet(3);
    check(out.str() == text, b + ": the stream operator prints the same");
}

/// display takes no input to check; with the checks off the text is the same.
void checks_and_their_switch() {
    const std::string on = EmptySet(2).display();
    ChecksScope off(false);
    check(EmptySet(2).display() == on, "display: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend(the_text_names_the_dimension);
    checks_and_their_switch();
    return test::finish("empty_set display");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
