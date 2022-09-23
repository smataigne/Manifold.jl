using LinearAlgebra, SkewLinearAlgebra, Test
import .Manifold as M

@testset "orthonormal.jl" begin
    for m in [1, 2, 10], n in [1, 2, 5], T in [Float64, Float32]
        if n<=m
            Q = M.randorthonormal(T, m, n)
            M.isorthonormal(Q)||display(Q)
            @test M.isorthonormal(Q) == true
            A = rand(T, m, n)
            @test M.isorthonormal(M.orthogonalize(A)) == true
        end
    end
end

@testset "stiefel.jl + exp/log.jl" begin
    for m in [2, 3, 5], n in [1, 2, 5], T in [Float64]
        if n <= m
            Q = M.randorthonormal(T, m, n)
            S = M.StiefelVector(Q)
            A = skewhermitian!(randn(n, n))
            B = randn(m - n, n)
            M.buildcomplement(S)
            Δ = M.TangentVector(A, B)
            E = exp(S, Δ)
            Δ₂ = M.gettangent(Δ, S)
            E₂ = exp(S, Δ₂)
            @test E ≈ E₂
            S2 = M.StiefelVector(M.randorthonormal(T, m , n))
            try
                V = log(S, S2)
                @test exp(S, V) ≈ S2.U
            catch
            end  
        end
    end
end



