@testitem "TOBSOptions: all keyword options on a small binary NLP" begin
    using NonconvexTOBS, HiGHS, Test
    # maximize sum(x) s.t. sum(x) ≤ 1, x ∈ {0,1}³ → one entry flipped, value 1.
    m = Model(x -> -sum(x))
    addvar!(m, zeros(3), ones(3))
    setinteger!(m, 1:3, [true, true, true])
    add_ineq_constraint!(m, x -> sum(x) - 1)
    options = TOBSOptions(;
        movelimit=0.5,
        pastN=5,
        convParam=0.05,
        constrRelax=0.5,
        timeLimit=5.0,
        optimizer=HiGHS.Optimizer,
        maxiter=50,
        timeStable=false,
        relaxDecay=0.5,
        maxInfeasible=3,
    )
    @test options.nt.movelimit == 0.5
    @test options.nt.pastN == 5
    @test options.nt.convParam == 0.05
    @test options.nt.constrRelax == 0.5
    @test options.nt.timeLimit == 5.0
    @test options.nt.optimizer === HiGHS.Optimizer
    @test options.nt.maxiter == 50
    @test options.nt.timeStable === false
    @test options.nt.relaxDecay == 0.5
    @test options.nt.maxInfeasible == 3
    r = optimize(m, TOBSAlg(), [0.0, 0.0, 0.0], options=options)
    @test r.minimum ≈ -1.0 atol = 1e-8
    @test sum(r.minimizer) ≈ 1.0 atol = 1e-8
    @test all(x -> x ≈ 0.0 || x ≈ 1.0, r.minimizer)
    @test r.error >= 0
end

@testitem "TOBSAlg: bounds must be 0/1" begin
    using NonconvexTOBS, Test
    m = Model(x -> sum(x .^ 2))
    addvar!(m, zeros(2), [1.0, 2.0])
    @test_throws ArgumentError optimize(m, TOBSAlg(), [0.0, 0.5])
end

@testitem "TOBSAlg: workspace and capabilities" begin
    using NonconvexTOBS, NonconvexCore, Test
    caps = solver_capabilities(TOBSAlg())
    @test caps.linear_ineq === true
    @test caps.nonlinear_ineq === true
    @test caps.linear_eq === false
    @test caps.nonlinear_eq === false
    @test caps.box === true
    @test caps.binary === true
    @test caps.continuous === false
    @test caps.integer === false
    m = Model(x -> -sum(x))
    addvar!(m, zeros(2), ones(2))
    add_ineq_constraint!(m, x -> sum(x) - 1)
    vm, x0v, _ = NonconvexCore.tovecmodel(m, [0.0, 0.0])
    w = Workspace(vm, TOBSAlg(), x0v; options=TOBSOptions(movelimit=0.5, timeLimit=5.0))
    @test w isa NonconvexTOBS.TOBSWorkspace
    @test w.options isa TOBSOptions
    @test w.options.nt.movelimit == 0.5
    r = NonconvexCore.optimize!(w)
    @test r.minimum ≈ -1.0 atol = 1e-8
end

@testitem "TOBSAlg: multiple linear inequality constraints" begin
    using NonconvexTOBS, Test
    # maximize x₁ + 2x₂ s.t. x₁ + x₂ ≤ 1, x₂ ≤ 1 → optimum x = (0, 1), value 2
    # (movelimit must allow at least one flip per iteration: 0.5 * 2 vars = 1)
    m = Model(x -> -(x[1] + 2x[2]))
    addvar!(m, zeros(2), ones(2))
    setinteger!(m, 1:2, [true, true])
    add_ineq_constraint!(m, x -> x[1] + x[2] - 1)
    add_ineq_constraint!(m, x -> x[2] - 1)
    options = TOBSOptions(movelimit=0.5, timeLimit=5.0)
    r = optimize(m, TOBSAlg(), [0.0, 0.0], options=options)
    @test r.minimum ≈ -2.0 atol = 1e-8
    @test r.minimizer ≈ [0.0, 1.0] atol = 1e-8
end

@testitem "TOBSAlg: optimize without x0 uses the model init" begin
    using NonconvexTOBS, Test
    # Regression test: optimize(m, TOBSAlg()) used to throw a MethodError
    # because no Workspace method without x0 existed.
    m = Model(x -> -sum(x))
    addvar!(m, zeros(2), ones(2))
    setinteger!(m, 1:2, [true, true])
    add_ineq_constraint!(m, x -> sum(x) - 1)
    r = optimize(m, TOBSAlg(), options=TOBSOptions(movelimit=0.5, timeLimit=5.0))
    @test r.minimum ≈ -1.0 atol = 1e-8
    @test sum(r.minimizer) ≈ 1.0 atol = 1e-8
    @test all(x -> x ≈ 0.0 || x ≈ 1.0, r.minimizer)
end

@testitem "TOBSAlg: degenerate pastN values do not crash" begin
    using NonconvexTOBS, Test
    # Regression test: pastN <= 0 used to give a BoundsError on the first
    # accepted step, and pastN >= maxiter used to leave the returned `error`
    # stuck at 1.0 because it was only computed when count > pastN.
    m = Model(x -> -sum(x))
    addvar!(m, zeros(2), ones(2))
    setinteger!(m, 1:2, [true, true])
    add_ineq_constraint!(m, x -> sum(x) - 1)
    for pastN in (0, -3)
        r = optimize(
            m,
            TOBSAlg(),
            [0.0, 0.0],
            options=TOBSOptions(
                pastN=pastN, movelimit=0.5, timeLimit=5.0, maxiter=5
            ),
        )
        @test r.minimum ≈ -1.0 atol = 1e-8
        @test sum(r.minimizer) ≈ 1.0 atol = 1e-8
    end
    r = optimize(
        m,
        TOBSAlg(),
        [0.0, 0.0],
        options=TOBSOptions(pastN=50, movelimit=0.5, timeLimit=5.0, maxiter=10),
    )
    @test r.minimum ≈ -1.0 atol = 1e-8
    @test r.error < 1.0
    @test r.error >= 0
end

@testitem "TOBSAlg: convergence error is a rolling mean of relative changes" begin
    using NonconvexTOBS, Test
    # Regression test: the convergence error used to be computed only after
    # count > pastN iterations with a running sum of raw objective values as
    # the denominator (which could even be negative for negative objectives).
    # It must now be a rolling mean of per-step relative objective changes,
    # detected as soon as two accepted steps exist.
    tobj = TraceFunction(x -> -sum(x))
    m = Model(tobj)
    addvar!(m, zeros(3), ones(3))
    setinteger!(m, 1:3, [true, true, true])
    add_ineq_constraint!(m, x -> sum(x) - 1)
    r = optimize(
        m,
        TOBSAlg(),
        zeros(3),
        options=TOBSOptions(pastN=20, movelimit=0.5, timeLimit=5.0, maxiter=50),
    )
    @test r.minimum ≈ -1.0 atol = 1e-8
    @test sum(r.minimizer) ≈ 1.0 atol = 1e-8
    @test r.error == 0.0
    @test !signbit(r.error) # never a negative "error"
    @test length(tobj.trace) < 15 # convergence detected well before maxiter
end

@testitem "TOBSAlg: infeasible subproblems are skipped without swallowing errors" begin
    using NonconvexTOBS, Test
    # Regression test: a failed MILP solve used to be papered over with a
    # blanket try/catch around the step application. A genuinely infeasible
    # subproblem (no primal solution) must be treated as a skipped step: the
    # relaxation logic kicks in and optimization returns normally.
    m = Model(x -> -sum(x))
    addvar!(m, zeros(2), ones(2))
    setinteger!(m, 1:2, [true, true])
    add_ineq_constraint!(m, x -> sum(x) - 1)
    # From x = (1, 1) the constraint sum(x) ≤ 1 is violated by 1, and with
    # constrRelax = 0.8 each subproblem demands sum(deltaX) ≤ -0.2 while the
    # move limit allows at most |deltaX| = 0.02 in total: every subproblem is
    # infeasible until the feasibility-restoration phase kicks in (which can
    # only take deltaX = 0 here, since any flip costs |deltaX| = 1 > 0.02).
    options = TOBSOptions(
        movelimit=0.01, constrRelax=0.8, maxInfeasible=5, maxiter=6, timeLimit=5.0
    )
    r = optimize(m, TOBSAlg(), [1.0, 1.0], options=options)
    @test r isa NonconvexTOBS.TOBSResult
    @test r.minimizer ≈ [1.0, 1.0] atol = 1e-8 # nothing bogus was applied
    @test r.minimum ≈ -2.0 atol = 1e-8
end

@testitem "TOBSAlg: continuous variables are rejected, not silently frozen" begin
    using NonconvexTOBS, Test
    # The design steps `deltaX` are `JuMP.Int`: a continuous variable at
    # x = 0.5 only permits integer moves in [-0.5, 0.5], i.e. {0}, so the
    # variable would be silently frozen. With `continuous=false` in the
    # capability declaration the pre-optimization capability check must
    # reject the model upfront instead.
    m = Model(x -> -sum(x))
    addvar!(m, zeros(2), ones(2))
    add_ineq_constraint!(m, x -> sum(x) - 1)
    @test_throws ArgumentError optimize(
        m, TOBSAlg(), [0.5, 0.5], options=TOBSOptions(timeLimit=5.0)
    )
    # a purely binary model and starting point is accepted
    setinteger!(m, 1:2, [true, true])
    r = optimize(m, TOBSAlg(), [0.0, 0.0], options=TOBSOptions(movelimit=0.5, timeLimit=5.0))
    @test r.minimum ≈ -1.0 atol = 1e-8
end
