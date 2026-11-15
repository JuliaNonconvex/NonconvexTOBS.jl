# NEWS

## 0.3.1

- Hardened step application and convergence tracking: a failed or unknown
  MILP solve is treated as a skipped step, and genuine exceptions propagate
  instead of being swallowed by a blanket try/catch.

## 0.3.0

- NonconvexCore 2 is now required with a strict capability declaration
  (equality constraints rejected, `infinite_bounds` declared); NonconvexUtils
  is no longer a dependency.
- DocumenterFragments docs with a binary-gradient-descent theory page; tests
  migrated to TestItems with JET, Aqua and formatting quality gates.

## 0.2.2

- Fixed equality-constraint handling (explicit rejection), the iteration
  count and diagnostics, with regression tests; coordinated the test
  dependencies with TopOpt 0.15 and NonconvexUtils 1.
