"""test_lean_empty_set - EmptySet against the CORALean oracle (cora.lean.EmptySet): exact answers"""
# ORACLE: contSet/@emptySet/and_.m
# ORACLE: contSet/@emptySet/center.m
# ORACLE: contSet/@emptySet/contains_.m
# ORACLE: contSet/@emptySet/convHull_.m
# ORACLE: contSet/@emptySet/empty.m
# ORACLE: contSet/@emptySet/emptySet.m
# ORACLE: contSet/@emptySet/interval.m
# ORACLE: contSet/@emptySet/isBounded.m
# ORACLE: contSet/@emptySet/isFullDim.m
# ORACLE: contSet/@emptySet/isIntersecting_.m
# ORACLE: contSet/@emptySet/isequal.m
# ORACLE: contSet/@emptySet/lift_.m
# ORACLE: contSet/@emptySet/not.m
# ORACLE: contSet/@emptySet/plus.m
# ORACLE: contSet/@emptySet/project.m
# ORACLE: contSet/@emptySet/projectHighDim_.m
# ORACLE: contSet/@emptySet/radius.m
# ORACLE: contSet/@emptySet/vertices_.m
# ORACLE: contSet/@emptySet/volume_.m

import unittest

import cora
import oracle_testing as oracle

DIMENSIONS = (1, 2, 5, 20)


def operands(rnd, n):
    """The other operands of dimension n: the empty set, the empty box, a box and a zonotope."""
    return [cora.EmptySet(n), cora.Interval.empty(n), rnd.box(n), rnd.zonotope(n, 3)]


def same_dimension(a, b):
    return oracle.check(a.dim == b.dim, True)


@oracle.needs_oracle
class Oracle(unittest.TestCase):
    def each(self, op, body):
        def run(rnd):
            for n in DIMENSIONS:
                body(rnd, n)
        oracle.run(op, run)

    def test_empty(self):
        def body(rnd, n):
            same_dimension(cora.EmptySet.empty(n), cora.lean.EmptySet.empty(n))
            same_dimension(cora.EmptySet(n), cora.lean.EmptySet.empty(n))  # the constructor
        self.each("emptySet.empty", body)

    def test_and_and_plus(self):
        def body(rnd, n):
            for S in operands(rnd, n):
                same_dimension(cora.EmptySet(n) & S, cora.lean.EmptySet(n).intersect(S))
        self.each("emptySet.and", body)

        def body(rnd, n):
            for S in operands(rnd, n):
                same_dimension(cora.EmptySet(n) + S, cora.lean.EmptySet(n).plus(S))
        self.each("emptySet.plus", body)

    def test_convex_hull(self):
        def body(rnd, n):
            I, Z = rnd.box(n), rnd.zonotope(n, 3)
            oracle.check(cora.EmptySet(n).convex_hull(I),
                         cora.lean.EmptySet(n).convex_hull(cora.lean.Interval(I)))
            # given back as it is: compared exactly in both modes (a certified containment is slow)
            oracle.check_exact("emptySet.convHull", cora.EmptySet(n).convex_hull(Z),
                               cora.lean.EmptySet(n).convex_hull(cora.lean.Zonotope(Z)))
        self.each("emptySet.convHull", body)

    def test_predicates(self):
        def contains(rnd, n):
            for S in operands(rnd, n):
                oracle.check(cora.EmptySet(n).contains(S), cora.lean.EmptySet(n).contains(S))
        self.each("emptySet.contains", contains)

        def meets(rnd, n):
            for S in operands(rnd, n):
                oracle.check(cora.EmptySet(n).is_intersecting(S),
                             cora.lean.EmptySet(n).is_intersecting(S))
        self.each("emptySet.isIntersecting", meets)

        def equal(rnd, n):
            for S in (cora.EmptySet(n), cora.Interval.empty(n)):
                oracle.check(cora.EmptySet(n).isequal(S), cora.lean.EmptySet(n).isequal(S))
        self.each("emptySet.isequal", equal)
        self.each("emptySet.isBounded", lambda rnd, n: oracle.check(
            cora.EmptySet(n).is_bounded(), cora.lean.EmptySet(n).is_bounded()))
        self.each("emptySet.isFullDim", lambda rnd, n: oracle.check(
            cora.EmptySet(n).is_full_dim(), cora.lean.EmptySet(n).is_full_dim()))

    def test_center_vertices_and_interval(self):
        # the oracle has no center and no vertex; its empty box is the crossed one (inf > sup)
        self.each("emptySet.center", lambda rnd, n: oracle.check(
            cora.EmptySet(n).center.shape[-1] == 0, cora.lean.EmptySet(n).center() is None))
        oracle.run("emptySet.vertices", lambda rnd: oracle.check(
            cora.EmptySet(2).vertices() == [], cora.lean.EmptySet(2).vertices().gather().shape[-1] == 0))

        def interval(rnd, n):
            lo, hi = (x.gather() for x in (cora.lean.EmptySet(n).interval().inf,
                                           cora.lean.EmptySet(n).interval().sup))
            mine = cora.EmptySet(n).interval()
            oracle.check(mine.dim == n and bool((lo > hi).all()), lo.shape[0] == n)
        self.each("emptySet.interval", interval)

    def test_lift_and_project(self):
        def lift(rnd, n):
            dims = [n + 1 - i for i in range(n)]
            same_dimension(cora.EmptySet(n).lift(n + 2, dims),
                           cora.lean.EmptySet(n).lift(n + 2, dims))
        self.each("emptySet.lift", lift)

        def project(rnd, n):
            dims = list(range(0, n, 2))
            same_dimension(cora.EmptySet(n).project(dims), cora.lean.EmptySet(n).project(dims))
        self.each("emptySet.project", project)

    def test_project_high_dim_radius_volume_and_complement(self):
        def high_dim(rnd, n):
            proj = [n + 1 - i for i in range(n)]
            same_dimension(cora.EmptySet(n).project_high_dim(n + 2, proj),
                           cora.lean.EmptySet(n).project_high_dim(n + 2, proj))
        self.each("emptySet.projectHighDim", high_dim)
        self.each("emptySet.radius", lambda rnd, n: oracle.check(
            cora.EmptySet(n).radius(), cora.lean.EmptySet(n).radius()))
        self.each("emptySet.volume", lambda rnd, n: oracle.check(
            cora.EmptySet(n).volume(), cora.lean.EmptySet(n).volume()))
        self.each("emptySet.not", lambda rnd, n: oracle.check(
            cora.EmptySet(n).complement(), cora.lean.EmptySet(n).complement()))


if __name__ == "__main__":
    unittest.main()
