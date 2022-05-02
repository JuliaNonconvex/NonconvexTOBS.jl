# NonconvexTOBS

[![CI](https://github.com/JuliaNonconvex/NonconvexTOBS.jl/workflows/CI/badge.svg?branch=main)](https://github.com/JuliaNonconvex/NonconvexTOBS.jl/actions?query=workflow%3ACI)
[![Coverage](https://codecov.io/gh/JuliaNonconvex/NonconvexTOBS.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JuliaNonconvex/NonconvexTOBS.jl)

The method of topological optimization of binary structures ([TOBS](https://www.sciencedirect.com/science/article/abs/pii/S0168874X17305619?via%3Dihub)) was originally developed in the context of optimal distribution of material in mechanical components. This package implements the heuristic for binary nonlinear programming problems.

## Example: use TOBS to optimize cantilever beam

Begin by installing the base package and what's necessary for the specific problem:

```julia
import Pkg
Pkg.add("NonconvexTOBS")
Pkg.add("TopOpt")
```

First, the finite element problem must be built. This is done using [TopOpt.jl](https://github.com/JuliaTopOpt/TopOpt.jl):

```julia
using NonconvexTOBS, TopOpt

E = 1.0 # Young’s modulus
v = 0.3 # Poisson’s ratio
f = 1.0 # downward force
rmin = 6.0 # filter radius
xmin = 0.001 # minimum density
V = 0.5 # maximum volume fraction
p = 3.0 # topological optimization penalty

# Define FEA problem
problem_size = (160, 100) # size of rectangular mesh
x0 = fill(1.0, prod(problem_size)) # initial design
problem = PointLoadCantilever(Val{:Linear}, problem_size, (1.0, 1.0), E, v, f)
```

FEA solver and auxiliary functions need to be defined as well:

```julia
solver = FEASolver(Direct, problem; xmin=xmin)
cheqfilter = DensityFilter(solver; rmin=rmin) # filter function
comp = TopOpt.Compliance(problem, solver) # compliance function
```

The usual topology optimization problem adresses compliance minimization under volume restriction. Therefore, the objective and the constraint are:

```julia
obj(x) = comp(cheqfilter(x)) # compliance objective
constr(x) = sum(cheqfilter(x)) / length(x) - V # volume fraction constraint
```

Finally, the optimization problem is defined and solved:

```julia
# Optimization setup
m = Model(obj) # create optimization model
addvar!(m, zeros(length(x0)), ones(length(x0))) # setup optimization variables
Nonconvex.add_ineq_constraint!(m, constr) # setup volume inequality constraint
options = TOBSOptions() # optimization options with default values
TopOpt.setpenalty!(solver, p)

# Perform TOBS optimization
@time r = Nonconvex.optimize(m, TOBSAlg(), x0; options=options)

# Results
@show obj(r.minimizer)
@show constr(r.minimizer)
topology = r.minimizer
```

Visualizing the results from this example:

![histories](https://user-images.githubusercontent.com/84910559/164938659-797a6a6d-3518-4f7b-a4ff-24b43b822080.png)

https://user-images.githubusercontent.com/84910559/164938672-e90adf1d-06f4-4a46-a956-d017c809d7dc.mp4

