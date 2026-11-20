# Options

The following are the options that can be set by passing them to [`TOBSOptions`](@ref), e.g. `TOBSOptions(movelimit = 0.1)`.
- `movelimit`: the maximum move limit in each iteration as a ratio of the total number of variables. Default value is 0.1, i.e. a maximum of 10% of the variables are allowed to flip value in each iteration.
- `convParam`: the tolerance value. The algorithm is said to have converged if the moving average of the relative change in the objective value in the last `pastN` iterations is less than `convParam`. Default value is 0.001.
- `pastN`: the number of past iterations used to compute the moving average of the relative change in the objective value. Default value is 20.
- `constrRelax`: the amount of constraint relaxation applied to the linear approximation in each iteration: a violated constraint must remove a fraction `1 - constrRelax` of its violation per iteration, while a satisfied constraint may grow by up to its current slack. Default value is 0.9.
- `timeLimit`: the time limit (in seconds) of each MILP solve for the linearized sub-problem. Default value is 1.0.
- `optimizer`: the `JuMP` optimizer type used to solve the MILP sub-problem. Default value is `HiGHS.Optimizer`.
- `maxiter`: the maximum number of iterations for the algorithm. Default value is 200.
- `timeStable`: a boolean value that when set to `true` switches on the time stability filter of the objective's gradient, discussed in the paper. Default value is `true`.
- `relaxDecay`: the factor shrinking an inflated `movelimit`/`constrRelax` back toward their default values after a successful step. Default value is 0.9.
- `maxInfeasible`: the number of consecutive infeasible subproblems before the algorithm switches to the feasibility-restoration phase, which minimizes the violated part of the linearized constraints instead of the objective. Default value is 5.
