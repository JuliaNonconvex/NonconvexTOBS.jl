using NonconvexTOBS, TopOpt, LinearAlgebra, Test

@testset "NonconvexTOBS.jl" begin
    E = 1.0 # Young’s modulus
    v = 0.3 # Poisson’s ratio
    f = 1.0 # downward force
    rmin = 4.0 # filter radius
    xmin = 0.0001 # minimum density
    problem_size = (60, 20)
    V = 0.5 # maximum volume fraction
    p = 4.0 # penalty
    x0 = fill(V, prod(problem_size)) # initial design

    problem = HalfMBB(Val{:Linear}, problem_size, (1.0, 1.0), E, v, f)

    solver = FEASolver(Direct, problem; xmin=xmin)
    cheqfilter = DensityFilter(solver; rmin=rmin)
    comp = TopOpt.Compliance(problem, solver)

    function obj(x)
        # minimize compliance
        return comp(cheqfilter(x))
    end
    function constr(x)
        # volume fraction constraint
        return sum(cheqfilter(x)) / length(x) - V
    end

    m = Model(obj)
    addvar!(m, zeros(length(x0)), ones(length(x0)))
    Nonconvex.add_ineq_constraint!(m, constr)

    options = TOBSOptions()
    TopOpt.setpenalty!(solver, p)
    # Method of Moving Asymptotes
    @time r = Nonconvex.optimize(m, TOBSAlg(), x0; options=options)

    @show obj(r.minimizer)
    @show constr(r.minimizer)
    topology = cheqfilter(r.minimizer)


end
