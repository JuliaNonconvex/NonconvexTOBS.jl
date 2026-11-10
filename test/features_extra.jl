@testitem "TOBSOptions: all keyword options on a small binary NLP" begin
    using NonconvexTOBS, HiGHS, Test
    # maximize sum(x) s.t. sum(x) ≤ 1, x ∈ {0,1}³ → one entry flipped, value 1.
    m = Model(x -> -sum(x))
    addvar!(m, zeros(3), ones(3))
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
    add_ineq_constraint!(m, x -> x[1] + x[2] - 1)
    add_ineq_constraint!(m, x -> x[2] - 1)
    options = TOBSOptions(movelimit=0.5, timeLimit=5.0)
    r = optimize(m, TOBSAlg(), [0.0, 0.0], options=options)
    @test r.minimum ≈ -2.0 atol = 1e-8
    @test r.minimizer ≈ [0.0, 1.0] atol = 1e-8
end
