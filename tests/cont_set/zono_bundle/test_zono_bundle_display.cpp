// test_zono_bundle_display - ZonoBundle display: the text that describes the bundle
//
// CORA: contSet/@zonoBundle/display.m
//
// CORA authors: Matthias Althoff
// Authors:       Claude Sonnet 5.5
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#include "zono_bundle_testing.h"

#include <sstream>

using namespace cora;
using test::check;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The text names the class, the dimension, the number of members and shows every member.
void text_of_a_bundle(const std::string &b) {
    const Zonotope Z(column({1.0, 2.0}), Tensor({{1.0}, {3.0}}));
    const ZonoBundle bundle({Z, Z});
    const std::string text = bundle.display();
    check(text.find("zonotope bundle") == 0, b + ": starts with the class");
    check(text.find("- dimension: 2") != std::string::npos &&
              text.find("- zonotopes: 2") != std::string::npos,
          b + ": dimension and number of members");
    check(text.find("- zonotope 2:") != std::string::npos &&
              text.find("    zonotope\n") != std::string::npos,
          b + ": every member, indented");
    std::ostringstream out;
    out << bundle;
    check(out.str() == text, b + ": the stream operator prints it");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend(text_of_a_bundle);
    // the display takes no input to get wrong; with the checks off the text is the same
    ChecksScope off(false);
    check(ZonoBundle::empty(1).display().find("zonotopes: 2") != std::string::npos,
          "display: the same with the checks off");
    return test::finish("zono_bundle display");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
