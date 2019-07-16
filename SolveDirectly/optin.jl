function Caloffmin(Gkx::Array{Int,1},Gky::Array{Int,1},m0::Float64,v0::Float64,
    lenGk::Int,b1::Array{Float64,1},b2::Array{Float64,1},optn::Int,mz::Float64)

    mat=zeros(ComplexF64,2*lenGk,2*lenGk)
    @inbounds for jj in 1:lenGk,ii in 1:lenGk
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

    @inbounds for mm in 1:lenGk
        vec_tmp=Gkx[mm]*b1+Gky[mm]*b2
        tmp=vec_tmp'*vec_tmp+v0
        mat[2*mm-1,2*mm-1]=tmp+mz
        mat[2*mm,2*mm]=tmp-mz
    end
    en_tmp,ev_tmp=eigen(Hermitian(mat))
    pt=partialsortperm(en_tmp,1:optn)
    return en_tmp[pt],ev_tmp[:,pt]
end

function fcoe(ind::Array{Int,2},lenind::Int,ev::Array{ComplexF64,2},optn::Int,
    guu::Float64,gdd::Float64,gud::Float64)
    mat_tmp=SharedArray{ComplexF64,1}(optn^4)
    myindex=Array{Int,2}(undef,4,optn^4)
    kk::Int=0
    @inbounds for ii in 1:optn,jj in 1:optn,mm in 1:optn,nn in 1:optn
        kk+=1
        myindex[:,kk].=[ii,jj,mm,nn]
    end
    @inbounds @sync @distributed for tt in 1:optn^4
        ii,jj,mm,nn=view(myindex,:,tt)
        uu::ComplexF64=dd::ComplexF64=ud::ComplexF64=0.0im
        @inbounds for kk in Base.OneTo(lenind)
            t1,t2,t3,t4=view(ind,:,kk)
            uu+=conj(ev[2*t1-1,ii]*ev[2*t2-1,jj])*ev[2*t3-1,mm]*ev[2*t4-1,nn]
            dd+=conj(ev[2*t1,ii]*ev[2*t2,jj])*ev[2*t3,mm]*ev[2*t4,nn]
            ud+=conj(ev[2*t1-1,ii]*ev[2*t2,jj])*ev[2*t3,mm]*ev[2*t4-1,nn]
            ud+=conj(ev[2*t1,ii]*ev[2*t2-1,jj])*ev[2*t3-1,mm]*ev[2*t4,nn]
        end
        mat_tmp[tt]=uu*guu+dd*gdd+ud*gud
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
    phase=abs(tmp[pt])/tmp[pt]
    tmp.=phase.*tmp
    for ii in 1:optn
        xopt[2*ii-1]=real(tmp[ii])
        xopt[2*ii]=imag(tmp[ii])
    end
    fopt=best_fitness(res)
    return xopt,fopt
end

function maincoe(Gkx::Array{Int,1},Gky::Array{Int,1},m0::Float64,v0::Float64,
    lenGk::Int,b1::Array{Float64,1},b2::Array{Float64,1},optn::Int,
    ind::Array{Int,2},lenind::Int,guu::Float64,gdd::Float64,gud::Float64,mz::Float64)

    en,ev=Caloffmin(Gkx,Gky,m0,v0,lenGk,b1,b2,optn,mz)
    coe=fcoe(ind,lenind,ev,optn,guu,gdd,gud)
    c4,c3,c1=decoe(coe,optn)
    Nstep=10^5
    xopt,fopt=optmin(optn,Nstep,c4,c3,c1,en)
    dxopt=[complex(xopt[2*ii-1],xopt[2*ii]) for ii in 1:optn]

    ev0=zeros(ComplexF64,2*lenGk)
    for ii in 1:optn
        ev0[:].+=ev[:,ii].*dxopt[ii]
    end

    u0::ComplexF64=0.0+0.0im
    @inbounds for ii in 1:lenind
        t1,t2,t3,t4=view(ind,:,ii)
        u0+=conj(ev0[2*t1-1]*ev0[2*t2-1])*ev0[2*t3-1]*ev0[2*t4-1]*guu
        u0+=conj(ev0[2*t1]*ev0[2*t2])*ev0[2*t3]*ev0[2*t4]*gdd
        u0+=conj(ev0[2*t1-1]*ev0[2*t2])*ev0[2*t3]*ev0[2*t4-1]*gud
        u0+=conj(ev0[2*t1]*ev0[2*t2-1])*ev0[2*t3-1]*ev0[2*t4]*gud
    end
    en0::Float64=0.0
    for ii in 1:optn
        en0+=abs2(dxopt[ii])*en[ii]
    end
    return real(u0)+en0/2,ev0
end
