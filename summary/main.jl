using Distributed
using DelimitedFiles,Dates
using JLD2, FileIO
using GR
nprocs()<2&&addprocs(4)
#addprocs(40)
@everywhere using BlackBoxOptim
@everywhere using LinearAlgebra
@everywhere using SharedArrays

include("Pexp.jl")
include("optin.jl")
include("func_feature.jl")

function main1d()
    println("---------mat-1D---------")
    t=time()
    Gmax,nb,optn=6,12,2
    pband=4
    v0,guu,gdd,gud=5.2,0.174868875,0.174868875,0.17446275 #guu=0.175275 gdd=0.17446275
    m0,mz=3.0,0.0
    nk1d=180
    lenb1,lenb2=sqrt(2),sqrt(2)
    b1,b2=[lenb1,0.0],[0.0,lenb2]

    Gkx,Gky,lenGk=CalGk(b1,b2,Gmax)
    mat=Caloffm(Gkx,Gky,m0,v0,lenGk)

    kx,ky,rr=linsym([0.0 0;lenb1/2 0; lenb1/2 lenb2/2;0 0],nk1d)

    #ben1d=enband1D(mat,Gkx,Gky,lenGk,kx,ky,b1,b2,v0,mz,nb)
    #return ben1d,rr
  
    ind,lenind=myidx(Gkx,Gky,lenGk)
    #ev0,u0=mainone(Gkx,Gky,m0,v0,lenGk,b1,b2,optn,ind,lenind,guu,gud,gdd,mz,pband)
    ph=pathout(Gmax,v0,m0,mz,optn,pband)
    #ev0,u0=mintwo(ph,lenGk,2)
    tmp=load("6-52-30-00-30-"*string(pband)*"/ev0.jld2")
    ev0=tmp["ev0"][:,1];u0=tmp["u0"]

    matH=zeros(ComplexF64,4*lenGk,4*lenGk)
    intBdgM!(matH,ev0,ind,lenind,lenGk,u0,guu,gud,gdd)
    println("matH_check:",sum(abs.(matH'-matH)))
    matHU0!(matH,mat,lenGk,v0,b1,b2,Gkx,Gky,mz,pband)
    
    ben1d=eigBdgM1D(matH,lenGk,Gkx,Gky,b1,b2,v0,nb,kx,ky,mz)
    return ben1d,rr
end
#=
ben1d,rr=main1d()
img=myplot(rr,ben1d)
println(ben1d[8,1]-ben1d[5,1])
println(ben1d[5,1]-ben1d[4,1])
println(ben1d[9,1]-ben1d[8,1])
=#

function main2d()
    println("----------- mat-2D -----------")
    t=time()
    Gmax,nb,optn=5,12,2
    pband=4
    mindx=2
    v0,m0,mz=5.2,3.0,0.0
    guu,gdd,gud=0.174868875,0.174868875,0.17446275 #guu=0.175275 gdd=0.17446275
    lenb1,lenb2=sqrt(2),sqrt(2)
    b1,b2=[lenb1,0.0],[0.0,lenb2]

    nk2d=5

    Gkx,Gky,lenGk=CalGk(b1,b2,Gmax)
    mat=Caloffm(Gkx,Gky,m0,v0,lenGk)
    kx,ky=bz2d([-lenb1/2 -lenb2/2;lenb1/2 lenb2/2],nk2d,0.0)
    println(length(kx))
    #ben2d,bev2d=enband2D(mat,Gkx,Gky,lenGk,kx,ky,b1,b2,v0,mz,nb)
    ind,lenind=myidx(Gkx,Gky,lenGk)

    ph=pathout(Gmax,v0,m0,mz,optn,pband)
    ev0,u0=mintwo(ph,lenGk,mindx)

    matH=zeros(ComplexF64,4*lenGk,4*lenGk)
    intBdgM!(matH,ev0,ind,lenind,lenGk,u0,guu,gud,gdd)
    println("matH_check:",sum(abs.(matH'-matH)))
    matHU0!(matH,mat,lenGk,v0,b1,b2,Gkx,Gky,mz,pband)

    ben2d,bev2d=eigBdgM2D(matH,lenGk,Gkx,Gky,b1,b2,v0,nb,kx,ky,mz)
    #return ben2d

    Chern,bcav=Bcuvat(bev2d)
    println(Chern)
    myexport(t,ben2d,bcav,Chern,guu,gud,gdd,ph,mindx)
    println(time()-t)
    return bcav
end
#ben2d=main2d()

function mainphase(n1::Int,n2::Int)
    println("---------phase---------")
    t=time()
    Gmax,nb,optn=6,12,2
    v0,guu,gdd,gud=5.2,0.174868875,0.174868875,0.17446275 #guu=0.175275 gdd=0.17446275
    m0,mz=3.0,0.0
    pband=n1

    lenb1,lenb2=sqrt(2),sqrt(2)
    b1,b2=[lenb1,0.0],[0.0,lenb2]

    Gkx,Gky,lenGk=CalGk(b1,b2,Gmax)
    mat=Caloffm(Gkx,Gky,m0,v0,lenGk)
   
    #ben1d=enband1D(mat,Gkx,Gky,lenGk,kx,ky,b1,b2,v0,mz,nb)
    #return ben1d,rr

    ind,lenind=myidx(Gkx,Gky,lenGk)
    myidx(Gkx,Gky,lenGk)
    #ev0,_=mainone(Gkx,Gky,m0,v0,lenGk,b1,b2,optn,ind,lenind,guu,gud,gdd,mz,pband)
    tmp=load("6-52-30-00-30-"*string(n1)*"/ev0.jld2","ev0")::Array{ComplexF64,2}
    ev0=tmp[:,n2]
    #gaugev0!(ev0)
    
    xx=collect(-pi:pi/200:pi)
    yy=collect(-pi:pi/200:pi)
    lenxx=length(xx)
    phs=wavedensity(xx,yy,[0.0,0.0],ev0,Gkx,Gky,lenGk,b1,b2)
    #phs=wavespin(xx,yy,[0.0,0.0],ev0,Gkx,Gky,lenGk,b1,b2)
    return phs
end
phs=mainphase(0,1) 

function mainminimum()
    println("---------min-two---------")
    Gmax,nb,optn=5,12,2
    pband=4
    v0,m0,mz=5.2,3.0,0.0
    guu,gdd,gud=0.174868875,0.174868875,0.17446275 #guu=0.175275 gdd=0.17446275
    lenb1,lenb2=sqrt(2),sqrt(2)
    b1,b2=[lenb1,0.0],[0.0,lenb2]

    Gkx,Gky,lenGk=CalGk(b1,b2,Gmax)
    ph=pathout(Gmax,v0,m0,mz,optn,pband)
  
    ind,lenind=myidx(Gkx,Gky,lenGk)
    maintwocoe(Gkx,Gky,m0,v0,lenGk,b1,b2,optn,ind,lenind,guu,gud,gdd,mz,ph,pband)
    maintwo(Gkx,Gky,m0,v0,lenGk,b1,b2,optn,ind,lenind,guu,gud,gdd,mz,ph)
end
#mainminimum()