// test_fullspace_display - fullspace display: the text that describes the whole space
//
// CORA: contSet/@fullspace/display.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/fullspace/fullspace.h"
#include "testing.h"

#include <sstream>

using namespace cora;
using test::check;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

void the_text_names_the_dimension(const std::string &b) {
    const std::string text = Fullspace(3).display();
    check(text.find("fullspace") == 0 && text.find("dimension: 3") != std::string::npos,
          b + ": class and dimension");
    std::ostringstream out;
    out << Fullspace(3);
    check(out.str() == text, b + ": the stream operator prints the same");
}

/// display takes no input to check; with the checks off the text is the same.
void checks_and_their_switch() {
    const std::string on = Fullspace(2).display();
    ChecksScope off(false);
    check(Fullspace(2).display() == on, "display: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend(the_text_names_the_dimension);
    checks_and_their_switch();
    return test::finish("fullspace display");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
