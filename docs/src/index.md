# NonconvexTOBS

![JuliaNonconvex logo](https://raw.githubusercontent.com/JuliaNonconvex/NonconvexTOBS.jl/main/docs/src/assets/logo.png)

## Description

The method of topology optimization of binary structures ([TOBS](https://www.sciencedirect.com/science/article/abs/pii/S0168874X17305619?via%3Dihub)) was originally developed in the context of optimal distribution of material in mechanical components. The TOBS algorithm only supports binary decision variables. The TOBS algorithm is a heuristic that relies on the sequential linearization of the objective and constraint functions, progressively enforcing the constraints in the process. The resulting binary linear program can be solved using any mixed integer linear programming (MILP) solver such as `HiGHS`. This process is repeated iteratively until convergence. This package implements the heuristic for binary nonlinear programming problems.

See [Construct an instance](construct.md) for the algorithm construction,
[Example](example.md) for a full topology-optimization example,
[Options](options.md) for the algorithm options and
[Docstrings](docstrings.md) for the full API reference.
