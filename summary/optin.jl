function Caloffmin(Gkx::Array{Int,1},Gky::Array{Int,1},m0::Float64,v0::Float64,
    lenGk::Int,b1::Array{Float64,1},b2::Array{Float64,1},optn::Int,mz::Float64,
    pband::Int)

    mat=zeros(ComplexF64,2*lenGk,2*lenGk)
    @inbounds for jj in 1:lenGk,ii in 1:lenGk
        t1 = Gkx[ii]-Gkx[jj]
        t2 = Gky[ii]-Gky[jj]
        if (t1==1&&t2==-1)||(t1==-1&&t2==1)||(t1==1&&t2==1)||(t1==-1&&t2==-1)
            mat[ii,jj]=mat[ii+lenGk,jj+lenGk] = v0/4.0
        end
        if t1==1 && t2==0
            mat[ii,jj+lenGk] = (-1.0-1.0im)*m0/4.0
            mat[ii+lenGk,jj] = (1.0-1.0im)*m0/4.0
        elseif t1==-1 && t2==0
            mat[ii,jj+lenGk] = (1.0+1.0im)*m0/4.0
            mat[ii+lenGk,jj] = (-1.0+1.0im)*m0/4.0
        elseif t1==0 && t2==1
            mat[ii,jj+lenGk] = (-1.0+1.0im)*m0/4.0
            mat[ii+lenGk,jj] = (1.0+1.0im)*m0/4.0
        elseif t1==0 && t2==-1
            mat[ii,jj+lenGk] = (1.0-1.0im)*m0/4.0
            mat[ii+lenGk,jj] = (-1.0-1.0im)*m0/4.0
        end
    end

    @inbounds for mm in 1:lenGk
        vec_tmp = Gkx[mm].*b1 .+Gky[mm].*b2
        tmp = dot(vec_tmp,vec_tmp)+v0
        mat[mm,mm] = tmp+mz
        mat[mm+lenGk,mm+lenGk] = tmp-mz
    end
    en_tmp,ev_tmp = eigen(Hermitian(mat))
    pt = partialsortperm(en_tmp,1+pband:optn+pband)
    return en_tmp[pt], ev_tmp[:,pt]
end

function fcoe(ind::Array{Int,2},lenind::Int,lenGk::Int,ev::Array{ComplexF64,2},
    optn::Int,guu::Float64,gud::Float64,gdd::Float64)

    mat_tmp = SharedArray{ComplexF64,1}(optn^4)
    myindex = Array{Int,2}(undef,4,optn^4)
    kk::Int=0
    @inbounds for ii in 1:optn,jj in 1:optn,mm in 1:optn,nn in 1:optn
        kk+=1
        myindex[:,kk].=[ii,jj,mm,nn]
    end

    @inbounds @sync @distributed for tt in 1:optn^4
        ii,jj,mm,nn=view(myindex,:,tt)
        uu::ComplexF64=dd::ComplexF64=ud::ComplexF64=0.0im
        @inbounds for kk in Base.OneTo(lenind)
            t1,t2,t3,t4 = view(ind,:,kk)
            uu+=conj(ev[t1,ii]*ev[t2,jj])*ev[t3,mm]*ev[t4,nn]
            dd+=conj(ev[t1+lenGk,ii]*ev[t2+lenGk,jj])*ev[t3+lenGk,mm]*ev[t4+lenGk,nn]
            ud+=conj(ev[t1,ii]*ev[t2+lenGk,jj])*ev[t3+lenGk,mm]*ev[t4,nn]
            ud+=conj(ev[t1+lenGk,ii]*ev[t2,jj])*ev[t3,mm]*ev[t4+lenGk,nn]
        end
        mat_tmp[tt] = uu*guu+dd*gdd+ud*gud
    end
    return Array(mat_tmp)
end

function decoe(coe::Array{ComplexF64,1},optn::Int)
    coem=Array{ComplexF64,4}(undef,optn,optn,optn,optn)
    kk::Int=0
    @inbounds for ii in 1:optn,jj in 1:optn,mm in 1:optn,nn in 1:optn
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
    return c4[1:k4],c3[1:k3],c1[1:k1]
end

function fcn2(c4::Array{ComplexF64,1},c3::Array{ComplexF64,1},
    c1::Array{ComplexF64,1},xx::Array{Float64,1},optn::Int,en::Array{Float64,1})

    vals=Array{ComplexF64,1}(undef,optn)
    dtmp::Float64=en0::Float64=0.0
    @inbounds for ii in 1:optn
        tmp=xx[2*ii-1]^2+xx[2*ii]^2
        en0+=tmp*en[ii]
        dtmp+=tmp
    end
    en0 = en0/dtmp
    dtmp=sqrt(dtmp)
    @inbounds for ii in 1:optn
        vals[ii]=complex(xx[2*ii-1]/dtmp,xx[2*ii]/dtmp)
    end

    res::ComplexF64=0.0im
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
    kk=0
    tmp::ComplexF64=0.0im
    @inbounds for ii in 1:optn
        for mm in 1:optn-1
            for nn in mm+1:optn
                kk+=1
                tmp+=c3[kk]*conj(vals[ii]^2)*vals[mm]*vals[nn]
            end
        end
    end
    res+=tmp+conj(tmp)

    kk=0
    @inbounds for ii in 1:optn
        for mm in 1:optn
            kk+=1
            res+=c1[kk]*conj(vals[ii]^2)*vals[mm]^2
        end
    end
    #
    return (en0+real(res))*1e6
end

function optmin(optn::Int,Nstep::Int,c4::Array{ComplexF64,1},
    c3::Array{ComplexF64,1},c1::Array{ComplexF64,1},en::Array{Float64,1})

    @inline foo2(xx::Array{Float64,1})=fcn2(c4,c3,c1,xx,optn,en)
    opt1=bbsetup(foo2;Method=:adaptive_de_rand_1_bin_radiuslimited,SearchRange=(-1.0,1.0),
            NumDimensions=2*optn,MaxFuncEvals=Nstep,TraceMode=:silent)
    res=bboptimize(opt1)
    xopt=normalize(best_candidate(res))

    tmp=Array{ComplexF64,1}(undef,optn)
    for ii in 1:optn
        tmp[ii]=complex(xopt[2*ii-1],xopt[2*ii])
    end
    abs(tmp[1])>abs(tmp[2]) ? pt=1 : pt=2
    println("pt=",pt," diff:",abs2(tmp[1])-abs2(tmp[2]))
    phase=abs(tmp[pt])/tmp[pt]
    tmp.=phase.*tmp

    fopt=best_fitness(res)::Float64
    return tmp, fopt, pt
end

function optcheck(xopt::Array{ComplexF64,1},lenGk::Int,optn::Int,guu::Float64,
    gud::Float64,gdd::Float64,en::Array{Float64,1},ev::Array{ComplexF64,2},
    lenind::Int,ind::Array{Int,2})

    ev0=zeros(ComplexF64,2*lenGk)
    for ii in 1:optn
        ev0.+=ev[:,ii].*xopt[ii]
    end

    u0::ComplexF64=0.0im
    @inbounds for ii in 1:lenind
        t1,t2,t3,t4 = view(ind,:,ii)
        u0 += conj(ev0[t1]*ev0[t2])*ev0[t3]*ev0[t4]*guu
        u0 += conj(ev0[t1+lenGk]*ev0[t2+lenGk])*ev0[t3+lenGk]*ev0[t4+lenGk]*gdd
        u0 += conj(ev0[t1]*ev0[t2+lenGk])*ev0[t3+lenGk]*ev0[t4]*gud
        u0 += conj(ev0[t1+lenGk]*ev0[t2])*ev0[t3]*ev0[t4+lenGk]*gud
    end
    en0::Float64 = 0.0
    for ii in 1:optn
        en0 += abs2(xopt[ii])*en[ii]
    end
    println("check:",u0+en0)
    return ev0,real(u0)+en0/2  
end

function gaugev0!(ev0::Array{ComplexF64,1})
    lenev=length(ev0)
    maxpt::Int=1
    maxtmp::Float64=abs2(ev0[1])
    for ii in 2:lenev
        if abs2(ev0[ii])>maxtmp
            maxpt=ii
            maxtmp=abs2(ev0[ii])
        end
    end
    phsfac::ComplexF64=ev0[maxpt]/abs(ev0[maxpt])
    for ii in 1:lenev
        ev0[ii]=ev0[ii]/phsfac
    end
    nothing
end


function mainone(Gkx::Array{Int,1},Gky::Array{Int,1},m0::Float64,v0::Float64,
    lenGk::Int,b1::Array{Float64,1},b2::Array{Float64,1},optn::Int,ind::Array{Int,2},
    lenind::Int,guu::Float64,gud::Float64,gdd::Float64,mz::Float64,pband::Int)

    en,ev=Caloffmin(Gkx,Gky,m0,v0,lenGk,b1,b2,optn,mz,pband)
    
    if abs(mz)<1e-10
        for ii in 1:round(Int,optn/2)
            ev[:,2*ii-1],ev[:,2*ii]=sigmazeig(ev[:,2*ii-1],ev[:,2*ii])
        end
    end

    coe=fcoe(ind,lenind,lenGk,ev,optn,guu,gud,gdd)
    c4,c3,c1=decoe(coe,optn)
    Nstep=10^5

    xopt,fopt,_=optmin(optn,Nstep,c4,c3,c1,en)
    println("xopt:")
    for ii in xopt
        println("  ",abs(ii))
    end
    println("fopt= ",fopt)
    ev0,u0=optcheck(xopt,lenGk,optn,guu,gud,gdd,en,ev,lenind,ind)
    gaugev0!(ev0)
    return ev0,u0
end


function maintwocoe(Gkx::Array{Int,1},Gky::Array{Int,1},m0::Float64,v0::Float64,
    lenGk::Int,b1::Array{Float64,1},b2::Array{Float64,1},optn::Int,ind::Array{Int,2},
    lenind::Int,guu::Float64,gud::Float64,gdd::Float64,mz::Float64,ph::String,pband::Int)
    
    en,ev=Caloffmin(Gkx,Gky,m0,v0,lenGk,b1,b2,optn,mz,pband)   
    if abs(mz)<1e-10
        for ii in 1:round(Int,optn/2)
            ev[:,2*ii-1],ev[:,2*ii]=sigmazeig(ev[:,2*ii-1],ev[:,2*ii])
        end
    end

    coe=fcoe(ind,lenind,lenGk,ev,optn,guu,gud,gdd)
    c4,c3,c1=decoe(coe,optn)
    save(ph*"/coe.jld2","en",en,"ev",ev,"c4",c4,"c3",c3,"c1",c1)
end
function loadcoe(ph::String)
    en=load(ph*"/coe.jld2","en")::Array{Float64,1}
    ev=load(ph*"/coe.jld2","ev")::Array{ComplexF64,2}
    c4=load(ph*"/coe.jld2","c4")::Array{ComplexF64,1}
    c3=load(ph*"/coe.jld2","c3")::Array{ComplexF64,1}
    c1=load(ph*"/coe.jld2","c1")::Array{ComplexF64,1}
    return en,ev,c4,c3,c1
end
function maintwo(Gkx::Array{Int,1},Gky::Array{Int,1},m0::Float64,v0::Float64,
    lenGk::Int,b1::Array{Float64,1},b2::Array{Float64,1},optn::Int,ind::Array{Int,2},
    lenind::Int,guu::Float64,gud::Float64,gdd::Float64,mz::Float64,ph::String)

    en,ev,c4,c3,c1=loadcoe(ph)
    Nstep=2*10^5

    ######  find the first minimum   #############
    xopt1,fopt,pt=optmin(optn,Nstep,c4,c3,c1,en)
    println("xopt:")
    for ii in xopt1
        println("  ",abs(ii))
    end
    println("fopt= ",fopt)
    ev1,u0=optcheck(xopt1,lenGk,optn,guu,gud,gdd,en,ev,lenind,ind)
    gaugev0!(ev1)

    ######  find the second minimum   #############
    xopt2=Array{ComplexF64,1}(undef,optn)
    for ii in 1:20
        xopt2,fopt,tmp=optmin(optn,Nstep,c4,c3,c1,en)
        if tmp!=pt
            break
        end
    end
    ev2,_=optcheck(xopt2,lenGk,optn,guu,gud,gdd,en,ev,lenind,ind)
    gaugev0!(ev2)

    ########   save the minimum   ##############
    ev0=Array{ComplexF64,2}(undef,2*lenGk,2)
    xopt=Array{ComplexF64,2}(undef,optn,2)
    if pt==1
        ev0[:,1].=ev1
        ev0[:,2].=ev2
        xopt[:,1].=xopt1
        xopt[:,2].=xopt2
    else
        ev0[:,1].=ev2
        ev0[:,2].=ev1
        xopt[:,1].=xopt2
        xopt[:,2].=xopt1
    end
    gaugev0!(ev0)
    save(ph*"/ev0.jld2","ev0",ev0,"xopt",xopt,"u0",u0)
end

###########################################
#           test part
############################################
function maincoetest(Gkx::Array{Int,1},Gky::Array{Int,1},m0::Float64,v0::Float64,
    lenGk::Int,b1::Array{Float64,1},b2::Array{Float64,1},optn::Int,
    ind::Array{Int,2},lenind::Int,guu::Float64,gud::Float64,gdd::Float64,mz::Float64)

    en,ev=Caloffmin(Gkx,Gky,m0,v0,lenGk,b1,b2,optn,mz)
    println("spin of single ev\n",sg(ev[:,1]),"\n",sg(ev[:,2]))
    println("v1|z|v2:\n",meanz(ev[:,1],ev[:,2]))
    println("v1|z|v1:\n",meanz(ev[:,1],ev[:,1]))
    println("v2|z|v2:\n",meanz(ev[:,2],ev[:,2]),"\n")
 #
    if abs(mz)<1e-10
        ev[:,1],ev[:,2]=sigmazilize(ev[:,1],ev[:,2])
        println("spin of single ev\n",sg(ev[:,1]),"\n",sg(ev[:,2]))
        println("v1|z|v2:\n",meanz(ev[:,1],ev[:,2]))
        println("v1|z|v1:\n",meanz(ev[:,1],ev[:,1]))
        println("v2|z|v2:\n",meanz(ev[:,2],ev[:,2]))
        println("v1*v2:\n",ev[:,1]'*ev[:,2])
        println("v1*v1:\n",ev[:,1]'*ev[:,1])
        println("v2*v2:\n",ev[:,2]'*ev[:,2])
    end
 #
    coe=fcoe(ind,lenind,lenGk,ev,optn,guu,gud,gdd)
    c4,c3,c1=decoe(coe,optn)
    Nstep=10^6
 #
    xopt,fopt=optmin(optn,Nstep,c4,c3,c1,en)
    println("fopt= ",fopt)
    ev0=zeros(ComplexF64,2*lenGk)
    en0::Float64 = 0.0

    for ii in 1:optn
        en0 += abs2(xopt[ii])*en[ii]
        ev0 .+=ev[:,ii].*xopt[ii]
    end
    u0::ComplexF64=0.0+0.0im
    @inbounds for ii in 1:lenind
        t1,t2,t3,t4 = view(ind,:,ii)
        u0 += conj(ev0[t1]*ev0[t2])*ev0[t3]*ev0[t4]*guu
        u0 += conj(ev0[t1+lenGk]*ev0[t2+lenGk])*ev0[t3+lenGk]*ev0[t4+lenGk]*gdd
        u0 += conj(ev0[t1]*ev0[t2+lenGk])*ev0[t3+lenGk]*ev0[t4]*gud
        u0 += conj(ev0[t1+lenGk]*ev0[t2])*ev0[t3]*ev0[t4+lenGk]*gud
    end

    return ev,ev0,xopt,en0,real(u0+en0/2)
end

function sigmazeig(ev1::Array{ComplexF64,1},ev2::Array{ComplexF64,1})
    pertu=Array{ComplexF64,2}(undef,2,2)

    pertu[1,1]=meanz(ev1,ev1)
    pertu[2,1]=meanz(ev2,ev1)
    pertu[1,2]=meanz(ev1,ev2)
    pertu[2,2]=meanz(ev2,ev2)
    
    en,ev=eigen(Hermitian(pertu))
    v1=ev[1,1].*ev1.+ev[2,1].*ev2
    v2=ev[1,2].*ev1.+ev[2,2].*ev2

    if en[1]<en[2]
        return v1,v2
    else
        return v2,v1
    end 
end