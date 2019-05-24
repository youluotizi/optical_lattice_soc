using LinearAlgebra
using DelimitedFiles
using Distributed
using JLD2, FileIO
cmds=true # true means calculate Coe, other means find opt
gg,fg=0.2,1.0
v0,m0=4.0,3.0
Gmax,optn=4,98
Nstep= 4*10^5
if cmds
    nprocs()==1&&addprocs(4)
    @everywhere using SharedArrays
else
    nprocs()==1&&addprocs(4)
    @everywhere using BlackBoxOptim
    @everywhere using LinearAlgebra
end
t=time()
lenb1=sqrt(2)
lenb2=sqrt(2)
b1,b2=[lenb1,0.0],[0.0,lenb2]

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
Gkx,Gky,lenGk=CalGk(b1,b2,Gmax)

function pathout(Gmax::Int,gg::Float64,fg::Float64,optn::Int)
    tmp=string(gg)
    ph=string(Gmax)*tmp[1]*tmp[3]
    tmp=string(fg)
    ph=ph*tmp[1]*tmp[3]*string(optn)
    return ph
end
ph=pathout(Gmax,gg,fg,optn)

function myind(Gkx::Array{Int,1},Gky::Array{Int,1},lenGk::Int,t::Float64)
    index0=Array{Int,2}(undef,4,lenGk^3*floor(Int,lenGk/2))
    ind0::Int=0
    @inbounds for ii in 1:lenGk,jj in 1:lenGk,mm in 1:lenGk
        tx=-Gkx[ii]-Gkx[jj]+Gkx[mm]
        ty=-Gky[ii]-Gky[jj]+Gky[mm]
        for nn in 1:lenGk
            if ty+Gky[nn]==0&&tx+Gkx[nn]==0
                ind0+=1
                index0[:,ind0].=[ii,jj,mm,nn]
                break
            end
        end
    end
    println(time()-t,"s index=",ind0)
    return index0[:,1:ind0],ind0
end
ind,lenind=myind(Gkx,Gky,lenGk,t)


function Caloffm(Gkx::Array{Int,1},Gky::Array{Int,1},m0::Float64,v0::Float64,lenGk::Int,
    b1::Array{Float64,1},b2::Array{Float64,1},optn::Int,t::Float64)
    mat=zeros(ComplexF64,2*lenGk,2*lenGk)
    for jj in 1:lenGk,ii in 1:lenGk
        t1=Gkx[ii]-Gkx[jj]
        t2=Gky[ii]-Gky[jj]
        if (t1==1&&t2==-1)||(t1==-1&&t2==1)||(t1==1&&t2==1)||(t1==-1&&t2==-1)
            mat[2*ii-1,2*jj-1]=mat[2*ii,2*jj]=v0/4.0
        end
        if t1==1&&t2==0
            mat[2*ii-1,2*jj]=(1.0-1.0im)*m0/4.0/1.0im
            mat[2*ii,2*jj-1]=(1.0+1.0im)*m0/4.0/1.0im
        elseif t1==-1&&t2==0
            mat[2*ii-1,2*jj]=(-1.0+1im)*m0/4.0/1im
            mat[2*ii,2*jj-1]=(-1.0-1im)*m0/4.0/1im
        elseif t1==0&&t2==1
            mat[2*ii-1,2*jj]=(1.0+1im)*m0/4.0/1im
            mat[2*ii,2*jj-1]=(1.0-1im)*m0/4.0/1im
        elseif t1==0&&t2==-1
            mat[2*ii-1,2*jj]=(-1.0-1im)*m0/4.0/1im
            mat[2*ii,2*jj-1]=(-1.0+1im)*m0/4.0/1im
        end
    end

    for mm in 1:lenGk
        vec_tmp=Gkx[mm]*b1+Gky[mm]*b2
        mat[2*mm-1,2*mm-1]=mat[2*mm,2*mm]=vec_tmp'*vec_tmp+v0
    end
    en_tmp,ev_tmp=eigen(Hermitian(mat))
    pt=partialsortperm(en_tmp,1:optn)

    println(time()-t,"s Gamma point")
    return en_tmp[pt],ev_tmp[:,pt]
end


function fcoe(ind::Array{Int,2},lenind::Int,ev::Array{ComplexF64,2},optn::Int,gg::Float64,fg::Float64,t::Float64)
    mat_tmp=SharedArray{ComplexF64,1}(optn^4)
    myindex=Array{Int,2}(undef,4,optn^4)
    kk::Int=0
    @inbounds for ii in 1:optn,jj in 1:optn,mm in 1:optn,nn in 1:optn
        kk+=1
        myindex[:,kk].=[ii,jj,mm,nn]
    end
    @inbounds @sync @distributed for tt in 1:optn^4
        ii,jj,mm,nn=view(myindex,:,tt)
        tmp::ComplexF64=0.0+0.0im
        @inbounds for kk in 1:lenind
            t1,t2,t3,t4=view(ind,:,kk)
            tmp+=conj(ev[2*t1-1,ii]*ev[2*t2-1,jj])*ev[2*t3-1,mm]*ev[2*t4-1,nn]
            tmp+=conj(ev[2*t1,ii]*ev[2*t2,jj])*ev[2*t3,mm]*ev[2*t4,nn]
            tmp+=conj(ev[2*t1-1,ii]*ev[2*t2,jj])*ev[2*t3,mm]*ev[2*t4-1,nn]*fg
            tmp+=conj(ev[2*t1,ii]*ev[2*t2-1,jj])*ev[2*t3-1,mm]*ev[2*t4,nn]*fg
        end
        mat_tmp[tt]=tmp*gg
    end
    println(time()-t," s Coecompleted")
    return Array(mat_tmp)
end
function decoe(coe::Array{ComplexF64,1},optn::Int)
    coem=Array{ComplexF64,4}(undef,optn,optn,optn,optn)
    kk::Int=0
    for ii in 1:optn,jj in 1:optn,mm in 1:optn,nn in 1:optn
        kk+=1
        coem[ii,jj,mm,nn]=coe[kk]
    end
    k4::Int=0
    c4=Array{ComplexF64,1}(undef,floor(Int,optn^4/4))
    for ii in 1:optn-1
        for jj in ii+1:optn
            for mm in 1:optn-1
                for nn in mm+1:optn
                    k4+=1
                    c4[k4]=coem[ii,jj,mm,nn]+coem[jj,ii,mm,nn]
                    c4[k4]+=coem[ii,jj,nn,mm]+coem[jj,ii,nn,mm]
                end
            end
        end
    end
    k3::Int=0
    c3=Array{ComplexF64,1}(undef,floor(Int,optn^3))
    for ii in 1:optn
        for mm in 1:optn-1
            for nn in mm+1:optn
                k3+=1
                c3[k3]=coem[ii,ii,mm,nn]+coem[ii,ii,nn,mm]
            end
        end
    end
    k2::Int=0
    c2=Array{ComplexF64,1}(undef,floor(Int,optn^3))
    for ii in 1:optn-1
        for jj in ii+1:optn
            for nn in 1:optn
                k2+=1
                c2[k2]=coem[ii,jj,nn,nn]+coem[jj,ii,nn,nn]
            end
        end
    end
    k1::Int=0
    c1=Array{ComplexF64,1}(undef,floor(Int,optn^2))
    for ii in 1:optn
        for mm in 1:optn
            k1+=1
            c1[k1]=coem[ii,ii,mm,mm]
        end
    end
    return c4[1:k4],c3[1:k3],c1[1:k1],k4,k3,k1
end


if cmds
    function maincoe(Gkx,Gky,m0,v0,lenGk,b1,b2,optn,ind,lenind,gg,fg,ph,t)
        en,ev=Caloffm(Gkx,Gky,m0,v0,lenGk,b1,b2,optn,t)
        coe=fcoe(ind,lenind,ev,optn,gg,fg,t)
        c4,c3,c1,lenc4,lenc3,lenc1=decoe(coe,optn)
        try
            mkdir(ph)
        catch
            nothing
        end
        @save ph*"/c4"*ph*".jld2" c4
        @save ph*"/c3"*ph*".jld2" c3
        @save ph*"/c1"*ph*".jld2" c1
        @save ph*"/ev"*ph*".jld2" ev
        @save ph*"/en"*ph*".jld2" en
        println("Coe complete ",time()-t)
        exit()
    end
    maincoe(Gkx,Gky,m0,v0,lenGk,b1,b2,optn,ind,lenind,gg,fg,ph,t)
else
    begin
        c4=load(ph*"/c4"*ph*".jld2","c4")
        c3=load(ph*"/c3"*ph*".jld2","c3")
        c1=load(ph*"/c1"*ph*".jld2","c1")
        ev=load(ph*"/ev"*ph*".jld2","ev")
        en=load(ph*"/en"*ph*".jld2","en")
        lenc4=size(c4,2)
        lenc3=size(c3,2)
        lenc1=size(c1,2)
    end
    for ii in workers()
    remotecall_fetch(()->(c4;c3;c1;en;lenc4;lenc3;lenc1;optn), ii)
    end
end

#=
function fcn(coe::Array{ComplexF64,1},xx::Array{Float64,1},optn::Int,en::Array{Float64,1})
    vals=Array{ComplexF64,1}(undef,2*optn)
    dtmp::Float64=en0::Float64=0.0
    for ii in 1:optn
        tmp=xx[2*ii-1]^2+xx[2*ii]^2
        en0+=tmp*en[ii]
        dtmp+=tmp
    end
    en0=en0/dtmp
    dtmp=sqrt(dtmp)
    for ii in 1:optn
        vals[ii]=complex(xx[2*ii-1]/dtmp,xx[2*ii]/dtmp)
    end
    res::ComplexF64=0.0+0.0im
    kk::Int=0
    for ii in 1:optn,jj in 1:optn,mm in 1:optn,nn in 1:optn
        kk+=1
        res+=coe[kk]*conj(vals[ii]*vals[jj])*vals[mm]*vals[nn]
    end
    #println(res+en0)
    return en0+real(res)
end
foo(xx::Array{Float64,1})=fcn(coe,xx,optn,en)
=#
@everywhere @inline function fcn2(c4::Array{ComplexF64,1},c3::Array{ComplexF64,1},
    c1::Array{ComplexF64,1},xx::Array{Float64,1},optn::Int,en::Array{Float64,1})
    vals=Array{ComplexF64,1}(undef,2*optn)
    dtmp::Float64=en0::Float64=0.0
    @inbounds for ii in 1:optn
        tmp=xx[2*ii-1]^2+xx[2*ii]^2
        en0+=tmp*en[ii]
        dtmp+=tmp
    end
    en0=en0/dtmp
    dtmp=sqrt(dtmp)
    @inbounds for ii in 1:optn
        vals[ii]=complex(xx[2*ii-1]/dtmp,xx[2*ii]/dtmp)
    end
    res::ComplexF64=0.0+0.0im
    kk::Int=0
    @inbounds for ii in Base.OneTo(optn-1)
        for jj in ii+1:optn
            for mm in Base.OneTo(optn-1)
                for nn in mm+1:optn
                    kk+=1
                    res+=c4[kk]*conj(vals[ii]*vals[jj])*vals[mm]*vals[nn]
                end
            end
        end
    end
    kk=0; tmp::ComplexF64=0.0+0.0im
    @inbounds for ii in 1:optn
        for mm in 1:optn-1
            for nn in mm+1:optn
                kk+=1
                tmp+=c3[kk]*conj(vals[ii]^2)*vals[mm]*vals[nn]
            end
        end
    end
    res+=tmp+conj(tmp)
    #=
    kk=0
    for ii in 1:optn-1
        for jj in ii+1:optn
            for nn in 1:optn
                kk+=1
                res+=c2[kk]*conj(vals[ii]*vals[jj])*vals[nn]^2
            end
        end
    end
    =#
    kk=0
    @inbounds for ii in 1:optn
        for mm in 1:optn
            kk+=1
            res+=c1[kk]*conj(vals[ii]^2)*vals[mm]^2
        end
    end
    #
    return en0+real(res)
end
@everywhere @inline foo2(xx::Array{Float64,1})=fcn2(c4,c3,c1,xx,optn,en)


function checkfopt(ev::Array{ComplexF64,2},en::Array{Float64,1},coe::Array{Float64,1},
    ind::Array{Int,2},lenind::Int,optn::Int,lenGk::Int,gg::Float64,fg::Float64)
    xopt=[complex(coe[2*ii-1],coe[2*ii]) for ii in 1:optn]
    xopt.=normalize(xopt)
    ev0=zeros(ComplexF64,2*lenGk)
    for ii in 1:optn
        ev0[:]+=ev[:,ii].*xopt[ii]
    end
    u0::ComplexF64=0.0+0.0im
    for ii in 1:lenind
        t1,t2,t3,t4=view(ind,:,ii)
        u0+=conj(ev0[2*t1-1]*ev0[2*t2-1])*ev0[2*t3-1]*ev0[2*t4-1]
        u0+=conj(ev0[2*t1]*ev0[2*t2])*ev0[2*t3]*ev0[2*t4]
        u0+=conj(ev0[2*t1-1]*ev0[2*t2])*ev0[2*t3]*ev0[2*t4-1]*fg
        u0+=conj(ev0[2*t1]*ev0[2*t2-1])*ev0[2*t3-1]*ev0[2*t4]*fg
    end
    en0::Float64=0.0
    for ii in 1:optn
        en0+=abs2(xopt[ii])*en[ii]
    end
    return u0*gg+en0
end
foocheck(coe::Array{Float64,1})=checkfopt(ev,en,coe,ind,lenind,optn,lenGk,gg,fg)


function testfcn(optn::Int)
    tol::Float64=0.0
    for ii in 1:20
        xxt=rand(Float64,2*optn)
        #println(foocheck(xxt))
        tol+=abs(foo2(xxt)-foocheck(xxt))
    end
    tol>1e-9&&exit()
    println("test_pass: ",tol)
    nothing
end
testfcn(optn)

# First run without any parallel procs used in eval
@everywhere function optmin(optn::Int,Nstep::Int)
    opt1 = bbsetup(foo2; Method=:adaptive_de_rand_1_bin_radiuslimited, SearchRange = (-1.0, 1.0),
            NumDimensions = 2*optn, MaxFuncEvals = Nstep,TraceMode=:silent)
    t=time()
    res = bboptimize(opt1)
    t=time()-t
    println(t,"s opt complete")
    xopt=normalize(best_candidate(res))
    tmp=Array{ComplexF64,1}(undef,optn)
    for ii in 1:optn
        tmp[ii]=complex(xopt[2*ii-1],xopt[2*ii])
    end
    abs(tmp[1])>abs(tmp[2]) ? pt=1 : pt=2
    phase=abs(tmp[pt])/tmp[pt]
    tmp.=phase.*tmp
    for ii in 1:optn
        xopt[2*ii-1]=real(tmp[ii])
        xopt[2*ii]=imag(tmp[ii])
    end
    fopt=best_fitness(res)
    println("fopt:",fopt)
    return xopt,fopt
end

#xopt,fopt=optmin(optn,Nstep)

function myprint(xopt,fopt,ph,t,Gmax,optn,gg,fg)
    dexopt=Array{Float64,2}(undef,optn,2)
    println("xotp:")
    for ii in 1:optn
        dexopt[ii,:]=xopt[2*ii-1:2*ii]
        println(xopt[2*ii-1],"  ",xopt[2*ii])
    end
    writedlm(ph*"/xopt"*ph,dexopt)
    writedlm(ph*"/fopt"*ph,["fotp",fopt,"time used",time()-t,"Gmax",Gmax,"optn",optn,"gg",gg,"fg",fg])
    println("fotp: ",fopt)
    println("Check:\n",foocheck(xopt))
    nothing
end
#myprint(xopt,fopt,ph,t,Gmax,optn,gg,fg)


para=[v0,m0,optn,gg,fg,Gmax]
function mainopt(optn::Int,Nstep::Int,ph::String,para::Array{Float64,1})
    nn=length(workers())
    dexopt=Array{Float64,2}(undef,optn,2)
    opt=pmap(ii->optmin(optn,Nstep),1:nn)
    paraname="v0,m0,optn,gg,fg,Gmax"
    for ii=1:nn
        xopt,fopt=opt[ii]
        for ii in 1:optn
            dexopt[ii,:]=xopt[2*ii-1:2*ii]
            println(xopt[2*ii-1],"  ",xopt[2*ii])
        end
        writedlm(ph*"/xopt"*ph*string(ii),dexopt)
        writedlm(ph*"/fopt"*ph*string(ii),["fotp",fopt,"time used",time()-t,paraname,para])
        println("fotp: ",fopt,"\n---------------")
        #println("Check:\n",foocheck(xopt))
    end
end
mainopt(optn,Nstep,ph,para)
#=
exit()
function comparev(optn,ev,lenGk,Nstep)
    xopt,_=optmin(optn,Nstep)
    tmp=Array{ComplexF64,1}(undef,optn)
    for ii in 1:optn
        tmp[ii]=complex(xopt[2*ii-1],xopt[2*ii])
    end
    ev0=zeros(ComplexF64,2*lenGk)
    for ii in 1:optn
        ev0[:].+=ev[:,ii].*tmp[ii]
    end
    vz::ComplexF64=0.0+0.0im
    for ii in lenGk
        vz+=conj(ev0[2*ii-1])*ev0[2*ii]+conj(ev0[2*ii])*ev0[2*ii-1]
    end
    return tmp,ev0
end
function compare2(optn,ev,lenGk,Nstep)
    ncom=10
    coe1=Array{ComplexF64,2}(undef,optn,ncom)
    vz=Array{ComplexF64,2}(undef,2*lenGk,optn)
    for ii in 1:ncom
        coe1[:,ii],vz[:,ii]=comparev(optn,ev,lenGk,Nstep)
    end
    return coe1,vz
end
com,vz=compare2(optn,ev,lenGk,Nstep)
=#
