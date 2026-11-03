// test_lean_interval_representsa - Interval::representsa against the CORALean oracle
//
// The oracle covers the types "emptySet", "point" and "origin" at tolerance 0, for nonempty boxes:
// it cannot take the empty one (bounds without columns), and its float "origin" and "point" are
// vacuously true there, where CORA says an empty set is neither. "emptySet" is checked against the
// sound oracle only: at dtype nearest the oracle reports the point [0.3, 0.3] as empty (its
// strict comparison has a margin there), which is no guarantee of anything.
//
// ORACLE: contSet/@interval/representsa_.m

#include "cora/cont_set/interval/interval.h"
#include "cora/global/oracle/interval.h"
#include "oracle_testing.h"

#include <string>

using namespace cora;

int main() {
    if (!test::oracle::available()) return 0;
    test::oracle::run("interval.representsa", [](test::oracle::Random &rnd) {
        const bool sound = test::oracle::mode() == test::oracle::Mode::sound;
        for (const int n : {1, 3}) {
            const Tensor c = rnd.vector(n);
            const std::vector<Interval> boxes{rnd.box(n), Interval(c, c),
                                              Interval(c.zeros_like(), c.zeros_like()),
                                              Interval(c.zeros_like(), c.abs())};
            for (const Interval &I : boxes)
                for (const char *type : {"emptySet", "point", "origin"})
                    if (std::string(type) != "emptySet" || sound)
                        test::oracle::check(I.representsa(type, 0),
                                            lean::Interval(I).representsa(type, 0));
        }
    });
    return test::finish("lean interval representsa");
}
