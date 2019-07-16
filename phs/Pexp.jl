using Distributed
using DelimitedFiles,Dates
using GR
#nprocs()<2&&addprocs(3)
#addprocs(40)
@everywhere using LinearAlgebra
@everywhere using BlackBoxOptim
@everywhere using SharedArrays

function CalGk(b1::Array{Float64,1},b2::Array{Float64,1},Gmax::Int)
    Gkx=Array{Int,1}(undef,(2*Gmax)^2)
    Gky=Array{Int,1}(undef,(2*Gmax)^2)
    kk::Int=0
    lenb=max(norm(b1),norm(b2))
    for jj in -Gmax:Gmax,ii in -Gmax:Gmax
        if norm(ii*b1+jj*b2)<Gmax*lenb+0.001
            kk+=1
            Gkx[kk]=ii; Gky[kk]=jj
        end
    end
    println("lenGk:",kk)
    return Gkx[1:kk],Gky[1:kk],kk
end

function myind(Gkx::Array{Int,1},Gky::Array{Int,1},lenGk::Int)
    ind=Array{Int,2}(undef,4,lenGk^3*floor(Int,lenGk/2))
    kk::Int=0
    @inbounds for ii in 1:lenGk,jj in 1:lenGk,mm in 1:lenGk
        tm::Int=-Gkx[ii]-Gkx[jj]+Gkx[mm]
        tn::Int=-Gky[ii]-Gky[jj]+Gky[mm]
        for nn in 1:lenGk
            if tn+Gky[nn]==0&&tm+Gkx[nn]==0
                kk+=1
                ind[:,kk]=[ii,jj,mm,nn]
                break
            end
        end
    end
    return ind[:,1:kk],kk
end

@everywhere include("/Users/zylt/Desktop/soc-jl/PhaseDiagram/optin.jl")
#@everywhere include("/Users/zylt/Desktop/soc-2/optin.jl")

function phsdiagm()
    t=time()
    Gmax,optn=5,2
    v0,guu,gdd,gud=2.0,0.174868875,0.174868875,0.17446275 #guu=0.175275 gdd=0.17446275

    lenb1,lenb2=sqrt(2),sqrt(2)
    b1,b2=[lenb1,0.0],[0.0,lenb2]

    Gkx,Gky,lenGk=CalGk(b1,b2,Gmax)
    ind,lenind=myind(Gkx,Gky,lenGk)
    #
    mz=collect(-0.0006:0.0006/100:0.0006)
    m0=collect(0.0:0.13/200:0.13)
    phs=SharedArray{ComplexF64,3}(3,length(mz),length(m0))
    lenm0=length(m0)

    for iz in 1:length(mz)
        @sync @distributed for i0 in 1:lenm0
            ev0=maincoe(Gkx,Gky,m0[i0],v0,lenGk,b1,b2,optn,ind,lenind,guu,gdd,gud,mz[iz])
            phs[:,iz,i0]=sg(ev0)
        end
    end
    println(time()-t,"s completed")
    return m0,phs
end
#m0,phs=phsdiagm()#;GR.heatmap(real(phs[3,:,:]))
#=
writedlm("phsx",real(phs[1,:,:]))
writedlm("phsy",real(phs[2,:,:]))
writedlm("phsz",real(phs[3,:,:]))
=#

function phsdiagmz()
    t=time()
    Gmax,optn=5,2
    v0,guu,gdd,gud=2.0,0.174868875,0.174868875,0.17446275 #guu=0.175275 gdd=0.17446275
    #guu,gdd,gud=1.0.*(guu,gdd,gud*0.99)
    lenb1,lenb2=sqrt(2),sqrt(2)
    b1,b2=[lenb1,0.0],[0.0,lenb2]

    Gkx,Gky,lenGk=CalGk(b1,b2,Gmax)
    ind,lenind=myind(Gkx,Gky,lenGk)

    mz=collect(-0.0006:0.001/20:0.0006)
    lenmz=length(mz)
    phs=SharedArray{ComplexF64,2}(3,lenmz)
    @sync @distributed for iz in 1:lenmz
        ev0=maincoe(Gkx,Gky,0.03,v0,lenGk,b1,b2,optn,ind,lenind,guu,gdd,gud,mz[iz])
        phs[:,iz]=sg(ev0)
    end
    return mz,phs
end
#m0,phs=phsdiagmz();
#GR.plot(real(phs[3,:]))
#writedlm("m0.txt",[m0 real(phs[3,:])])

function phsdiagm0()
    t=time()
    Gmax,optn=5,2
    v0,guu,gdd,gud=2.0,0.174868875,0.174868875,0.17446275 #guu=0.175275 gdd=0.17446275
    lenb1,lenb2=sqrt(2),sqrt(2)
    b1,b2=[lenb1,0.0],[0.0,lenb2]

    Gkx,Gky,lenGk=CalGk(b1,b2,Gmax)
    ind,lenind=myind(Gkx,Gky,lenGk)

    println(time()-t,"s 1")
    m0=collect(0.0:1.0:20.0) #mz=0/; 0.000406125
    lenm0=length(m0)
    phs=SharedArray{ComplexF64,2}(3,lenm0)
    println(time()-t,"s 2")
    @sync @distributed for i0 in 1:lenm0
        ev0=maincoe(Gkx,Gky,m0[i0],v0,lenGk,b1,b2,optn,ind,lenind,guu,gdd,gud,0.0)
        phs[:,i0]=sg(ev0)
    end
    println(time()-t,"s end")
    return m0,phs
end
m0,phs=phsdiagm0()
GR.plot(real(phs[3,:]))

function phsdiagtest()
    t=time()
    Gmax,optn=5,10
    v0,guu,gdd,gud=2.0,0.174868875,0.174868875,0.17446275 #guu=0.175275 gdd=0.17446275
    lenb1,lenb2=sqrt(2),sqrt(2)
    b1,b2=[lenb1,0.0],[0.0,lenb2]

    Gkx,Gky,lenGk=CalGk(b1,b2,Gmax)
    ind,lenind=myind(Gkx,Gky,lenGk)

    maincoe(Gkx,Gky,.03,v0,lenGk,b1,b2,optn,ind,lenind,guu,gdd,gud,0.003)
end
#phsdiagtest()
