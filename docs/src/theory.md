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

## TOBS as binary gradient descent

The sequential linearization above is more than a heuristic recipe: it is the
**binary analogue of gradient descent**, and viewing it this way explains
both the algorithm's behaviour and its tuning parameters.

### The linear minimization oracle

Gradient descent moves from $x^k$ along the direction that best decreases
the *linear model* of $f$,

$$d^k = \arg\min_{\lVert d\rVert \le \alpha} \; \nabla f(x^k)^\top d \;=\; -\alpha\,\nabla f(x^k),$$

and the step size $\alpha$ guards the trust region of the linearization.
On the binary domain the same principle applies, except the step must stay
binary: the subproblem

$$\min_{x \in \{0,1\}^n} \; \nabla f(x^k)^\top (x - x^k) \quad \text{s.t.} \quad
\lVert x - x^k \rVert_1 \le \delta$$

is the exact **linear minimization oracle** over the discrete domain — the
closest possible analogue of the negative-gradient step. For an
*unconstrained* binary problem its solution is explicit: flip exactly the
variables with the most negative gradient components, up to the move
budget $\delta$. The **move limit is the step size**: with $\delta$ too
large the linearization is trusted far outside its validity region and the
iterate overshoots — the discrete counterpart of divergence under a
large $\alpha$; with $\delta$ too small the method crawls. The adaptive
schedule (inflate on infeasible subproblems, decay after successful steps)
is therefore nothing but a step-size controller for a binary walk.

### Constraints as the projection

Projected gradient descent solves
$\min_{\lVert d\rVert \le \alpha} \nabla f^\top d$ *subject to the linearized
constraints*. The TOBS subproblem does exactly that with a MILP solver —
the discrete analogue of the projection step — while the relaxation
parameter $\bar g_j$ plays the role of the feasibility tolerance that keeps
the projection well-defined when the linearized feasible set is empty.
Accepting a step only when it decreases the *true* violation is the same
safeguard projected-gradient methods use to avoid drifting outside the
constraint set under nonlinear curvature.

### What the implementation generalizes

The literature method (Sivapuram, Picelli & Xie, 2018) was derived for
structural topology optimization: minimize compliance subject to a volume
constraint, with the sensitivities of one mechanical model. This package
restates it as a generic **binary nonlinear programming** method:

- the objective and constraints are arbitrary smooth functions,
  differentiated automatically (NonconvexCore), so the "sensitivities" of
  any discipline are just gradients;
- any number of constraints is handled, with per-iteration relaxation and
  the feasibility-restoration phase instead of a single hand-tuned volume
  bound;
- the MILP subproblem (move limit + relaxation) is assembled purely from
  first-order information, so the same code applies to binary programs far
  outside topology optimization.

In gradient-descent language: the literature method is one instance of
binary gradient descent with a specific objective, one constraint and a
fixed step-size rule; this implementation provides the general oracle-based
scheme, with an adaptive step size and a feasibility controller.

## Strengths and limitations

- The MILP subproblem is solved to global optimality by a branch-and-bound
  solver (HiGHS), so each step is the *best* binary step inside the move
  limit — a strong property for highly nonconvex binary problems.
- The method is a heuristic: there is no guarantee of global optimality of
  the fixed point, and the quality of the solution depends on the initial
  design, the move limit schedule and the relaxation parameters.
- The linearization requires first-order information only; no Hessians are
  needed.
