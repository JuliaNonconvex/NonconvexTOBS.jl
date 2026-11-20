# Example

In this example, the classic topology optimization problem of minimizing the compliance of the structure subject to a volume constraint. Begin by installing and loading the packages required.

```julia
using NonconvexTOBS, Nonconvex
using Pkg
Pkg.add("TopOpt")
using TopOpt
```

Define the problem and its parameters using [TopOpt.jl](https://github.com/JuliaTopOpt/TopOpt.jl).

```julia
E = 1.0 # Young’s modulus
v = 0.3 # Poisson’s ratio
f = 1.0 # downward force
rmin = 6.0 # filter radius
xmin = 0.001 # minimum density
V = 0.5 # maximum volume fraction
p = 3.0 # SIMP penalty

# Define FEA problem
problem_size = (160, 100) # size of rectangular mesh
x0 = fill(1.0, prod(problem_size)) # initial design
problem = PointLoadCantilever(problem_size, (1.0, 1.0), E, v, f; celltype=:Linear)
solver = FEASolver(DirectSolver, problem; xmin=xmin)
cheqfilter = DensityFilterFun(solver; rmin=rmin) # filter function
comp = TopOpt.ComplianceFun(solver) # compliance function
```

Define the objective and constraint functions.

```julia
obj(x) = comp(cheqfilter(PseudoDensities(x))) # compliance objective
constr(x) = sum(cheqfilter(PseudoDensities(x))) / length(x) - V # volume fraction constraint
```

Finally, define the optimization problem using `Nonconvex.jl` and optimize it.

```julia
tobj = TraceFunction(obj, on_grad=true)
m = Model(tobj)
addvar!(m, zeros(length(x0)), ones(length(x0)))
add_ineq_constraint!(m, constr)
TopOpt.setpenalty!(solver, p)
options = TOBSOptions()

r = optimize(m, TOBSAlg(), x0; options=options)
r.minimizer
r.minimum
```

The following is a visualization of the optimization history using this example.

![histories](https://user-images.githubusercontent.com/84910559/164938659-797a6a6d-3518-4f7b-a4ff-24b43b822080.png)

![gif](https://user-images.githubusercontent.com/19524993/167059067-f08502a8-c62d-4d62-a2df-e132efc5e25c.gif)
