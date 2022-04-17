using NonconvexTOBS, TopOpt, LinearAlgebra, Test, GLMakie

@testset "Example" begin
    E = 1.0 # Young’s modulus
    v = 0.3 # Poisson’s ratio
    f = 1.0 # downward force
    rmin = 6.0 # filter radius
    xmin = 0.001 # minimum density
    V = 0.5 # maximum volume fraction
    p = 3.0 # penalty

    # problem_size = (60, 20)
    # x0 = fill(1.0, prod(problem_size)) # initial design
    # problem = HalfMBB(Val{:Linear}, problem_size, (1.0, 1.0), E, v, f)

    global problem_size = (160, 100)
    x0 = fill(1.0, prod(problem_size)) # initial design
    problem = PointLoadCantilever(Val{:Linear}, problem_size, (1.0, 1.0), E, v, f)

    solver = FEASolver(Direct, problem; xmin=xmin)
    cheqfilter = DensityFilter(solver; rmin=rmin)
    comp = TopOpt.Compliance(problem, solver)

    obj(x) = comp(cheqfilter(x)) # compliance objective

    constr(x) = sum(cheqfilter(x)) / length(x) - V # volume fraction constraint

    m = Model(obj)
    addvar!(m, zeros(length(x0)), ones(length(x0)))
    Nonconvex.add_ineq_constraint!(m, constr)

    options = TOBSOptions(;
        constrRelax = 0.1,
        movelimit = 0.1,
        timeStable = true,
        timeLimit = 1.0
    )
    TopOpt.setpenalty!(solver, p)
    # TOBS
    @time r = Nonconvex.optimize(m, TOBSAlg(), x0; options=options)

    @show obj(r.minimizer)
    @show constr(r.minimizer)
    global topology = r.minimizer
end

function dispQuad(nelx,nely,vec)
    # nelx = number of elements along x axis (number of columns in matrix)
    # nely = number of elements along y axis (number of lines in matrix)
    # vec = vector of scalars, each one associated to an element.
      # this vector is already ordered according to element IDs
    quadVec=zeros(nely,nelx)
    for i in 1:nely
      for j in 1:nelx
        quadVec[nely-(i-1),j] = vec[(i-1)*nelx+1+(j-1)]
      end
    end
    display(heatmap(1:nelx,1:nely,quadVec'))
end

dispQuad(problem_size...,topology)