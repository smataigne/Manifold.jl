@views  function Base.log(S₁::StiefelVector{T, METRIC}, S₂::StiefelVector{T, METRIC}) where {T,METRIC}
    ϵ = eps(T)*1000
    N = 50
    δt = 1/(N-1)
    t = Array(0:δt:1)
    γ = norm(S₁.U - S₂.U)
    Δ = projection(S₂.U, S₁)
    Δ ./= norm(Δ)
    Δ .*= γ
    Δˢ = similar(Δ, size(Δ, 1), size(Δ, 2))
    itermax = 100; iter = 0
    while γ > ϵ  && iter < itermax
        Δˢ .= Matrix(exp(S₁, Δ))
        Δˢ .-= S₂.U
        γ = norm(Δˢ)
        for j = N:-1:1
            Δˢ = projection(Δˢ, StiefelVector(exp(S₁, Δ.*t[j])))
            Δˢ ./= norm(Δˢ)
            Δˢ .*= γ
        end
        Δ .-= Δˢ
        iter +=1
        #display(Δ)
    end
    return Δ
end