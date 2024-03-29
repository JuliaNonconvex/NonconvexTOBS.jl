using NonconvexTOBS, TopOpt, Test, NonconvexUtils

@testset "Example" begin

    # Recreating cantilever problem from original paper

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

    # define FEA solver and auxiliary functions
    solver = FEASolver(Direct, problem; xmin = xmin)
    cheqfilter = DensityFilter(solver; rmin = rmin) # filter function
    comp = TopOpt.Compliance(solver) # compliance function

    obj(x) = comp(cheqfilter(PseudoDensities(x))) # compliance objective
    constr(x) = sum(cheqfilter(PseudoDensities(x))) / length(x) - V # volume fraction constraint

    # Optimization setup
    tobj = TraceFunction(obj, on_grad = true)
    m = Model(tobj) # create optimization model
    addvar!(m, zeros(length(x0)), ones(length(x0))) # setup optimization variables
    Nonconvex.add_ineq_constraint!(m, constr) # setup volume inequality constraint
    options = TOBSOptions() # optimization options with default values
    TopOpt.setpenalty!(solver, p)

    # Perform TOBS optimization
    @time r = Nonconvex.optimize(m, TOBSAlg(), x0; options = options)

    # Results
    @show obj(r.minimizer)
    @show constr(r.minimizer)
    topology = r.minimizer

end
