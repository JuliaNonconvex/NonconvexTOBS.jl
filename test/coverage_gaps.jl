@testitem "Input validation throws" begin
    using NonconvexTOBS, NonconvexCore, LinearAlgebra, Test

    # Bounds must be binary.
    m = NonconvexCore.Model(x -> sum(x .^ 2))
    addvar!(m, [0.0, 0.0], [1.0, 2.0])
    vm = NonconvexCore.tovecmodel(m)[1]
    w = NonconvexTOBS.Workspace(vm, TOBSAlg(), [0.5, 0.5])
    @test_throws ArgumentError NonconvexTOBS.optimize!(w)

    # Equality constraints are not supported.
    m2 = NonconvexCore.Model(x -> sum(x .^ 2))
    addvar!(m2, [0.0, 0.0], [1.0, 1.0])
    add_eq_constraint!(m2, x -> x[1] - x[2])
    vm2 = NonconvexCore.tovecmodel(m2)[1]
    w2 = NonconvexTOBS.Workspace(vm2, TOBSAlg(), [0.5, 0.5])
    @test_throws ArgumentError NonconvexTOBS.optimize!(w2)
end
