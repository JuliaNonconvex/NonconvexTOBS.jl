# Theory: sequential binary linearization (TOBS)

## The binary nonlinear program

`TOBSAlg` targets problems of the form

$$\min_{x} \; f(x) \quad \text{s.t.} \quad g(x) \le 0, \quad x_i \in \{0, 1\},$$

where $f$ and $g$ are smooth but possibly nonconvex functions of the binary
design variables. Exhaustive search is exponential in the number of
variables, so the TOBS heuristic (Sivapuram, Picelli & Xie, 2018) replaces
the nonlinear binary program with a *sequence* of binary linear programs
that are cheap for a MILP solver.

## Sequential linearization

At the current iterate $x^k$, the objective and every constraint are
linearized around $x^k$:

$$f(x) \approx f(x^k) + \nabla f(x^k)^\top (x - x^k), \qquad
g_j(x) \approx g_j(x^k) + \nabla g_j(x^k)^\top (x - x^k).$$

The next iterate solves the binary linear program

$$\min_{x} \; \nabla f(x^k)^\top x \quad \text{s.t.} \quad
g_j(x^k) + \nabla g_j(x^k)^\top (x - x^k) \le \bar g_j, \quad
x_i \in \{0,1\}, \quad \lVert x - x^k \rVert_1 \le \delta,$$

where $\delta$ is the **move limit** and $\bar g_j$ is a constraint
relaxation parameter. The move limit keeps the linearization trustworthy
(the linear model is only locally valid), while the relaxation
`constrRelax` keeps the MILP feasible while the constraints are being
enforced progressively. Both parameters adapt across iterations: they are
inflated when a subproblem is infeasible (giving the MILP more freedom and
slack) and decayed geometrically (`relaxDecay`) after successful steps.

## Constraint enforcement

The linearized constraints are only approximations, so a step accepted by
the MILP can violate the *true* nonlinear constraints. The algorithm
therefore measures the true constraint violation `norm(model.ineq_constraints(x))`
after every step and accepts a step only when it reduces the true
infeasibility; otherwise the step is skipped and the subproblem is relaxed
further (feasibility restoration). This mirrors an augmented-Lagrangian
view of feasibility: the relaxation parameter plays the role of a penalty
weight that is raised only when progress stalls.

## Convergence and stopping

The iteration is a fixed-point process $x^{k+1} = T(x^k)$ on the discrete
design space. Convergence is declared by a **stability criterion**: the
objective history over the last `pastN` iterations is monitored, and the
algorithm stops once the design has stopped improving (or the iteration
limit is reached). The gradients of the nonlinear model are computed with
automatic differentiation through NonconvexCore.

## Strengths and limitations

- The MILP subproblem is solved to global optimality by a branch-and-bound
  solver (HiGHS), so each step is the *best* binary step inside the move
  limit — a strong property for highly nonconvex binary problems.
- The method is a heuristic: there is no guarantee of global optimality of
  the fixed point, and the quality of the solution depends on the initial
  design, the move limit schedule and the relaxation parameters.
- The linearization requires first-order information only; no Hessians are
  needed.
