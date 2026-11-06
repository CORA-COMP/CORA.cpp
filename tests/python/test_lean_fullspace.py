"""test_lean_fullspace - Fullspace against the CORALean oracle (its fullspace.* operations)

Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
Written:       03-November-2026
Last update:   04-November-2026
Reviewed:      ---
"""
# ORACLE: contSet/@fullspace/Inf.m
# ORACLE: contSet/@fullspace/and_.m
# ORACLE: contSet/@fullspace/box.m
# ORACLE: contSet/@fullspace/center.m
# ORACLE: contSet/@fullspace/contains_.m
# ORACLE: contSet/@fullspace/convHull_.m
# ORACLE: contSet/@fullspace/enclosePoints.m
# ORACLE: contSet/@fullspace/fullspace.m
# ORACLE: contSet/@fullspace/interval.m
# ORACLE: contSet/@fullspace/isBounded.m
# ORACLE: contSet/@fullspace/isFullDim.m
# ORACLE: contSet/@fullspace/isIntersecting_.m
# ORACLE: contSet/@fullspace/isequal.m
# ORACLE: contSet/@fullspace/lift_.m
# ORACLE: contSet/@fullspace/not.m
# ORACLE: contSet/@fullspace/plus.m
# ORACLE: contSet/@fullspace/project.m
# ORACLE: contSet/@fullspace/projectHighDim_.m
# ORACLE: contSet/@fullspace/radius.m
# ORACLE: contSet/@fullspace/volume_.m

import unittest

import cora
import oracle_testing as oracle


# -----------------------------------------  BEGIN CODE  ----------------------------------------- #


# ============================================  MAIN  ============================================ #

@oracle.needs_oracle
class Oracle(unittest.TestCase):
    """Each run is exact on the lean backend at nearest and consistent with the sound oracle."""

    def test_sets(self):
        def body(rnd):
            for n in (1, 5):
                F, L = cora.Fullspace(n), cora.lean.Fullspace(n)
                oracle.check(cora.Fullspace.inf(n), cora.lean.Fullspace.inf(n))
                oracle.check(F.box(), L.box())
                oracle.check(F.interval(), L.interval())
                oracle.check(F.plus(cora.Fullspace(n)), L.plus(cora.lean.Fullspace(n)))
                oracle.check(F.convex_hull(cora.Fullspace(n)), L.convex_hull(cora.lean.Fullspace(n)))
                oracle.check(F.intersect(cora.Fullspace(n)), L.intersect(cora.lean.Fullspace(n)))
                I = rnd.box(n)
                oracle.check(F.intersect(I), L.intersect(cora.lean.Interval(I)))
                oracle.check(F.lift(n + 2, list(range(n))), L.lift(n + 2, list(range(n))))
                oracle.check(F.center, L.center())
        oracle.run("fullspace sets", body)

    def test_predicates(self):
        def body(rnd):
            for n in (1, 5):
                F, L = cora.Fullspace(n), cora.lean.Fullspace(n)
                I = rnd.box(n)
                # predicates of the full space
                oracle.check(F.contains(I), L.contains(cora.lean.Interval(I)))
                oracle.check(F.is_intersecting(I), L.is_intersecting(cora.lean.Interval(I)))
                oracle.check(F.isequal(cora.Fullspace(n)), L.isequal(cora.lean.Fullspace(n)))
                oracle.check(F.is_bounded(), L.is_bounded())
                oracle.check(F.is_full_dim(), L.is_full_dim())
                oracle.check(cora.Fullspace.enclose_points(rnd.matrix(n, 3)),
                             cora.lean.Fullspace.enclose_points(rnd.matrix(n, 3)))
        oracle.run("fullspace predicates", body)

    def test_projections_measures_and_complement(self):
        def body(rnd):
            F, L = cora.Fullspace(3), cora.lean.Fullspace(3)
            oracle.check(F.project([2, 0]), L.project([2, 0]))
            # at N = n CORA and CORALean agree; for a larger N CORALean gives R^N, CORA refuses
            oracle.check(F.project_high_dim(3, [1, 0, 2]), L.project_high_dim(3, [1, 0, 2]))
            # R^0 is left out: CORA answers NaN for the radius and the volume, CORALean Inf
            for n in (1, 2, 5):
                oracle.check(cora.Fullspace(n).radius(), cora.lean.Fullspace(n).radius())
                oracle.check(cora.Fullspace(n).volume(), cora.lean.Fullspace(n).volume())
            for n in (0, 1, 5):
                oracle.check(cora.Fullspace(n).complement().dim == n,
                             cora.lean.Fullspace(n).complement().dim == n)
        oracle.run("fullspace projections", body)


if __name__ == "__main__":
    unittest.main()

# ----------------------------------------  END OF CODE  ----------------------------------------- #
