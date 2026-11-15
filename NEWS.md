# NEWS

## 0.3.3

- Solver settings now use the MOI-generic `JuMP.set_silent`/`JuMP.set_time_limit_sec`, so the pluggable `optimizer` option is honored.\n- Non-finite constraint values fail fast with a clear `ArgumentError` instead of a NaN MILP rhs.
## 0.3.2

- Fixed the feasibility-restoration phase: the slack constraint now prices
  the remaining positive violation (`violSlack >= c + J·dx` instead of the
  reversed sign) and the phase drops the relaxation schedule (which is what
  made the normal subproblem infeasible), so it actually finds the
  violation-reducing flips. Added a regression test and this NEWS file.

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
