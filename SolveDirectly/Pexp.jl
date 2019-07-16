using Distributed
using SharedArrays
using DelimitedFiles,Dates
using JLD2, FileIO
using BlackBoxOptim
using GR
nprocs()<2&&addprocs(3)
@everywhere using LinearAlgebra


function pathout(Gmax::Int,gg::Float64,fg::Float64,optn::Int,ch::Int)
    tmp=string(gg)
    ph=string(Gmax)*tmp[1]*tmp[3]
    tmp=string(fg)
    ph=ph*tmp[1]*tmp[3]*string(optn)*string(ch)
    return ph
end

####################################################
#         the point in reciprocal lattice vector
####################################################
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


####################################################
#                   Zones
####################################################
function linsym(plist::Array{Float64,2},num::Int)
    lenp=size(plist,1)
    p=transpose(plist)
    lenpath::Float64=0.0
    for ii in 1:lenp-1
        lenpath+=norm(p[:,ii+1]-p[:,ii])
    end
    delta=lenpath/(num-1)
    rr=Float64[0.0]
    r0::Float64=0.0
    bz=Array{Float64,2}(undef,2,num+20)
    kk=1; bz[:,1]=p[:,1]
    for ii in 1:lenp-1
        p0=p[:,ii]; vc=p[:,ii+1].-p0
        lvc=norm(vc); vc.=vc./lvc
        for jj in 1:num
            if jj*delta>lvc-0.2*delta
                kk+=1; bz[:,kk]=p[:,ii+1]
                r0+=lvc-(jj-1)*delta
                push!(rr,r0)
                break
            end
            kk+=1
            bz[:,kk]=p0+jj*delta*vc
            r0+=delta
            push!(rr,r0)
        end
    end
    return bz[1,1:kk],bz[2,1:kk],rr
end
function symline(pointlist::Array{Float64,2},num::Int)
    lenlist=size(pointlist,1)
    linbz=Array{Float64,2}(undef,2,(num-1)*(lenlist-1)+lenlist)
    points=pointlist'
    linbz[:,1]=points[:,1]
    kk=2
    for ii in 1:lenlist-1
        vec_tmp=points[:,ii+1]-points[:,ii]
        vec_tmp=vec_tmp./num
        for jj in 1:num
            linbz[:,kk]=points[:,ii]+jj*vec_tmp
            kk+=1
        end
        linbz[:,kk-1]=points[:,ii+1]
    end
    return linbz[1,:],linbz[2,:]
end
function bz2d(pointlist::Array{Float64,2},num::Int,span::Float64)
    xtmp,_=symline(pointlist,num,t)
    tt=ceil(Int,num*span/2)
    lenx=length(xtmp)
    bzx=Array{Float64,1}(undef,lenx+2*tt)
    bzx[tt+1:tt+lenx].=xtmp
    lstart=xtmp[1]
    rstart=xtmp[end]
    delta=xtmp[2]-xtmp[1]
    for ii in 1:tt
        bzx[tt+1-ii]=lstart-ii*delta
        bzx[tt+lenx+ii]=rstart+ii*delta
    end
    return bzx[:],bzx[:]
end

####################################################
#               offdiag of initial matrix
####################################################
function Caloffm(Gkx::Array{Int,1},Gky::Array{Int,1},m0::Float64,
    v0::Float64,lenGk::Int)
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
            mat[2*ii-1,2*jj]=(-1.0+1.0im)*m0/4.0/1.0im
            mat[2*ii,2*jj-1]=(-1.0-1.0im)*m0/4.0/1.0im
        elseif t1==0&&t2==1
            mat[2*ii-1,2*jj]=(1.0+1.0im)*m0/4.0/1.0im
            mat[2*ii,2*jj-1]=(1.0-1.0im)*m0/4.0/1.0im
        elseif t1==0&&t2==-1
            mat[2*ii-1,2*jj]=(-1.0-1.0im)*m0/4.0/1.0im
            mat[2*ii,2*jj-1]=(-1.0+1.0im)*m0/4.0/1.0im
        end
    end
    return mat
end

####################################################
#            single particle energy band
####################################################
function enband2D(mat::Array{ComplexF64,2},Gkx::Array{Int,1},Gky::Array{Int,1},
    lenGk::Int,kx::Array{Float64,1},ky::Array{Float64,1},b1::Array{Float64,1},
    b2::Array{Float64,1},v0::Float64,mz::Float64,nb::Int)

    lenky=length(ky)
    lenkx=length(kx)
    en=Array{Float64,3}(undef,nb,lenkx,lenky)

    for iky in 1:lenky,ikx in 1:lenkx
        @inbounds for mm in 1:lenGk
            vec_tmp=[kx[ikx],ky[iky]]+Gkx[mm]*b1+Gky[mm]*b2
            tmp=vec_tmp'*vec_tmp+v0
            mat[2*mm-1,2*mm-1]=tmp+mz
            mat[2*mm,2*mm]=tmp-mz
        end
        en_tmp=eigvals(Hermitian(mat))
        pt=partialsortperm(en_tmp,1:nb)
        en[:,ikx,iky]=en_tmp[pt]
    end
    return en
end

function enband1D(mat::Array{ComplexF64,2},Gkx::Array{Int,1},Gky::Array{Int,1},
    lenGk::Int,kx::Array{Float64,1},ky::Array{Float64,1},b1::Array{Float64,1},
    b2::Array{Float64,1},v0::Float64,mz::Float64,nb::Int)

    lenkx=length(kx)
    en=Array{Float64,2}(undef,nb,lenkx)
    ev=Array{ComplexF64,3}(undef,2*lenGk,nb,lenkx)

    for ik in 1:lenkx
        @inbounds for mm in 1:lenGk
            vec_tmp=[kx[ik],ky[ik]]+Gkx[mm]*b1+Gky[mm]*b2
            tmp=vec_tmp'*vec_tmp+v0
            mat[2*mm-1,2*mm-1]=tmp+mz
            mat[2*mm,2*mm]=tmp-mz
        end
        en_tmp=eigvals(Hermitian(mat))
        pt=partialsortperm(en_tmp,1:nb)
        en[:,ik]=en_tmp[pt]
    end
    return en#,ev
end

####################################################
#      the index result to phi-4 not integrated to zero
####################################################
function myind(Gkx::Array{Int,1},Gky::Array{Int,1},lenGk::Int)
    index0=Array{Int,2}(undef,4,lenGk^3*floor(Int,lenGk/2))
    ind0::Int=0
    @inbounds for ii in 1:lenGk,jj in 1:lenGk,mm in 1:lenGk
        tm::Int=-Gkx[ii]-Gkx[jj]+Gkx[mm]
        tn::Int=-Gky[ii]-Gky[jj]+Gky[mm]
        for nn in 1:lenGk
            if tn+Gky[nn]==0&&tm+Gkx[nn]==0
                ind0+=1
                index0[:,ind0].=[ii,jj,mm,nn]
                break
            end
        end
    end
    return index0[:,1:ind0],ind0
end

####################################################
#          bound state energy and u0
####################################################
function u0en0(ind::Array{Int,2},lenind::Int,mat::Array{ComplexF64,2},lenGk::Int,
    Gkx::Array{Int,1},Gky::Array{Int,1},b1::Array{Float64,1},b2::Array{Float64,1},
    v0::Float64,guu::Float64,gdd::Float64,gud::Float64,nb::Int,ph::Int)

    for mm in 1:lenGk
        vec_tmp=Gkx[mm]*b1+Gky[mm]*b2
        mat[2*mm-1,2*mm-1]=mat[2*mm,2*mm]=dot(vec_tmp,vec_tmp)+v0
    end
    en_tmp,ev_tmp=eigen(Hermitian(mat))
    pt=argmin(en_tmp)
    ev0=ev_tmp[:,pt]
    u0::ComplexF64=0.0+0.0im
    for ii in 1:lenind
        t1,t2,t3,t4=view(ind,:,ii)
        u0+=conj(ev0[2*t1-1]*ev0[2*t2-1])*ev0[2*t3-1]*ev0[2*t4-1]*guu
        u0+=conj(ev0[2*t1]*ev0[2*t2])*ev0[2*t3]*ev0[2*t4]*gdd
        u0+=conj(ev0[2*t1-1]*ev0[2*t2])*ev0[2*t3]*ev0[2*t4-1]*gud
        u0+=conj(ev0[2*t1]*ev0[2*t2-1])*ev0[2*t3-1]*ev0[2*t4]*gud
    end
    abs(imag(u0))>1e-8&&println("u0 error")
    return real(u0)+en_tmp[pt]/2,ev0
end

#####################################################
#           the interaction part of Bdg matrix
#####################################################
function intBdgM(lenind::Int,ind::Array{Int,2},ev0::Array{ComplexF64,1},
    lenGk::Int,u0::Float64,guu::Float64,gdd::Float64,gud::Float64)
    lenmat::Int=2*lenGk
    mat=zeros(ComplexF64,2*lenmat,2*lenmat)
    @inbounds for ii in 1:lenind
        t1,t2,t3,t4=view(ind,:,ii)
        #
        tmp::ComplexF64=2*conj(ev0[2*t2-1])*ev0[2*t3-1]*guu
        tmp+=conj(ev0[2*t2])*ev0[2*t3]*gdd
        mat[2*t1-1,2*t4-1]+=tmp
        mat[2*t1-1+lenmat,2*t4-1+lenmat]+=conj(tmp)
        #
        tmp=conj(ev0[2*t2])*ev0[2*t3-1]*gud
        mat[2*t1-1,2*t4]+=tmp
        mat[2*t1-1+lenmat,2*t4+lenmat]+=conj(tmp)

        tmp=conj(ev0[2*t2-1])*ev0[2*t3]*gud
        mat[2*t1,2*t4-1]+=tmp
        mat[2*t1+lenmat,2*t4-1+lenmat]+=conj(tmp)

        tmp=2*conj(ev0[2*t2])*ev0[2*t3]*gdd+conj(ev0[2*t2-1])*ev0[2*t3-1]*guu
        mat[2*t1,2*t4]+=tmp
        mat[2*t1+lenmat,2*t4+lenmat]+=conj(tmp)
        #
        ############# off diag #####################
        tmp=ev0[2*t3-1]*ev0[2*t4-1]*guu
        mat[2*t1-1,2*t2-1+lenmat]+=tmp
        mat[2*t1-1+lenmat,2*t2-1]+=conj(tmp)
        #mat[2*t1-1+lenmat,2*t2-1]+=conj(tmp)

        tmp=ev0[2*t3]*ev0[2*t4-1]*gud
        mat[2*t1-1,2*t2+lenmat]+=tmp
        mat[2*t1+lenmat,2*t2-1]+=conj(tmp)
        #mat[2*t1+lenmat,2*t2-1]+=conj(tmp)

        tmp=ev0[2*t3-1]*ev0[2*t4]*gud
        mat[2*t1,2*t2-1+lenmat]+=tmp
        mat[2*t1-1+lenmat,2*t2]+=conj(tmp)
        #mat[2*t1-1+lenmat,2*t2]+=conj(tmp)

        tmp=ev0[2*t3]*ev0[2*t4]*gdd
        mat[2*t1,2*t2+lenmat]+=tmp
        mat[2*t1+lenmat,2*t2]+=conj(tmp)
        #
    end
    return mat.-Diagonal(fill(u0,2*lenmat))
end

#####################################################
#               the Bdg matrix
#####################################################
function matH1!(matH::Array{ComplexF64,2},mat::Array{ComplexF64,2},lenGk::Int,
    v0::Float64,b1::Array{Float64,1},b2::Array{Float64,1},Gkx::Array{Int,1},
    Gky::Array{Int,1},mz::Float64)
    lenmat=2*lenGk
    tauz=Diagonal([ones(ComplexF64,lenmat); -ones(ComplexF64,lenmat)])
    @inbounds for ii in 1:2*lenGk
        mat[ii,ii]=0.0
    end

    matH[1:lenmat,1:lenmat]+=mat./2
    matH[lenmat+1:end,lenmat+1:end]+=conj.(mat)./2

    mtmp=copy(matH)
    @inbounds for mm in 1:lenGk
        tmp=Gkx[mm]*b1+Gky[mm]*b2
        stmp=tmp'*tmp+v0
        mtmp[2*mm-1,2*mm-1]+=(stmp+mz)/2
        mtmp[2*mm,2*mm]+=(stmp-mz)/2

        tmp=Gkx[mm]*b1+Gky[mm]*b2
        stmp=tmp'*tmp+v0
        mtmp[2*mm-1+lenmat,2*mm-1+lenmat]+=(stmp+mz)/2
        mtmp[2*mm+lenmat,2*mm+lenmat]+=(stmp-mz)/2
    end
    ben=eigvals(mtmp)
    u0=minimum(real(ben))
    @inbounds for mm in 1:2*lenmat
        matH[mm,mm]-=u0
    end
    lmul!(tauz,matH)
end

@everywhere function BdgM2d(matH0::Array{ComplexF64,2},matH::Array{ComplexF64,2},
    lenGk::Int,b1::Array{Float64,1},b2::Array{Float64,1},v0::Float64,
    Gkx::Array{Int,1},Gky::Array{Int,1},kx::Float64,ky::Float64,nb::Int)

    lenmat=2*lenGk
    @inbounds for mm in 1:lenGk
        tmp=[kx,ky]+Gkx[mm]*b1+Gky[mm]*b2
        stmp=tmp'*tmp+v0
        matH[2*mm-1,2*mm-1]+=(stmp+mz)/2
        matH[2*mm,2*mm]+=(stmp-mz)/2

        tmp=[-kx,-ky]+Gkx[mm]*b1+Gky[mm]*b2
        stmp=tmp'*tmp+v0
        matH[2*mm-1+lenmat,2*mm-1+lenmat]-=(stmp+mz)/2
        matH[2*mm+lenmat,2*mm+lenmat]-=(stmp-mz)/2
    end
    ben,bev=eigen(matH)
    pt=partialsortperm(real(ben),1:lenmat+nb)
    return ben[pt[lenmat+1-nb:end]],bev[:,pt[lenmat+1-nb:end]]
end

@everywhere function BdgM1d(matH::Array{ComplexF64,2},lenGk::Int,
    b1::Array{Float64,1},b2::Array{Float64,1},v0::Float64,Gkx::Array{Int,1},
    Gky::Array{Int,1},kx::Float64,ky::Float64,nb::Int,mz::Float64)

    lenmat=2*lenGk
    @inbounds for mm in 1:lenGk
        tmp=[kx,ky]+Gkx[mm]*b1+Gky[mm]*b2
        stmp=tmp'*tmp+v0
        matH[2*mm-1,2*mm-1]+=(stmp+mz)/2
        matH[2*mm,2*mm]+=(stmp-mz)/2

        tmp=[-kx,-ky]+Gkx[mm]*b1+Gky[mm]*b2
        stmp=tmp'*tmp+v0
        matH[2*mm-1+lenmat,2*mm-1+lenmat]-=(stmp+mz)/2
        matH[2*mm+lenmat,2*mm+lenmat]-=(stmp-mz)/2
    end
    ben=eigvals(matH)
    pt=partialsortperm(real(ben),1:lenmat+nb)
    return ben[pt[lenmat+1-nb:end]]
end

#####################################################
#              eigen Bdg
#####################################################
function eigBdgM1D(ind::Array{Int,2},lenind::Int,mat::Array{ComplexF64,2},lenGk::Int,
    Gkx::Array{Int,1},Gky::Array{Int,1},b1::Array{Float64,1},b2::Array{Float64,1},
    guu::Float64,gdd::Float64,gud::Float64,v0::Float64,nb::Int,kx::Array{Float64,1},
    ky::Array{Float64,1},u0::Float64,ev0::Array{ComplexF64,1},mz::Float64)
    println("---test---")
    t=time()
    lenkx=length(kx)
    ben=Array{ComplexF64,2}(undef,2*nb,lenkx)
    matH=intBdgM(lenind,ind,ev0,lenGk,guu,gdd,gud,u0)
    println(time()-t)
    matH1!(matH,mat,lenGk,v0,b1,b2,Gkx,Gky,mz)
    println(time()-t)
    ben_tmp=pmap((ix,iy)->BdgM1d(matH[:,:],lenGk,b1,b2,v0,Gkx,Gky,ix,iy,nb,mz),kx,ky)
    for ikx in 1:lenkx
        ben[:,ikx]=ben_tmp[ikx]
    end
    println(time()-t,"\n---end--")
    return ben
end

function eigBdgM2D(ind::Array{Int,2},lenind::Int,mat::Array{ComplexF64,2},lenGk::Int,
    Gkx::Array{Int,1},Gky::Array{Int,1},b1::Array{Float64,1},b2::Array{Float64,1},
    gg::Float64,v0::Float64,nb::Int,u0::Float64,ev0::Array{ComplexF64,1},
    fg::Float64,kx::Array{Float64,1},ky::Array{Float64,1},ph::String,mz::Float64)

    matH=intBdgM(lenind,ind,ev0,lenGk,guu,gdd,gud,u0)
    matH1!(matH,mat,lenGk,v0,b1,b2,Gkx,Gky,mz)
    lenkx=length(kx)
    lenky=length(ky)
    ben=Array{ComplexF64,3}(undef,2*nb,lenkx,lenky)
    bev=Array{ComplexF64,4}(undef,4*lenGk,2*nb,lenkx,lenky)
    kxlist=Array{Float64,1}(undef,lenkx*lenky)
    kylist=Array{Float64,1}(undef,lenkx*lenky)
    kk::Int=0
    for iky in 1:lenky,ikx in 1:lenkx
        kk+=1
        kylist[kk]=ky[iky]
        kxlist[kk]=kx[ikx]
    end
    ben_tmp=pmap((ix,iy)->BdgM2d(mat,matH,lenGk,b1,b2,v0,Gkx,Gky,ix,iy,nb),kxlist,kylist)
    kk=0
    for iky in 1:lenky,ikx in 1:lenkx
        kk+=1
        ben[:,ikx,iky],bev[:,:,ikx,iky]=ben_tmp[kk]
    end
    return ben,bev
end

#####################################################
#             Berry curvature and Chern number
#####################################################
function Bcuvat(bev::Array{ComplexF64,4})
    lenev,mid,lenkx,lenky=size(bev)
    mid=Int(mid/2)
    lenev=Int(lenev/2)
    Chern=Array{Float64,1}(undef,mid)
    bcav=Array{Float64,3}(undef,mid,lenkx-1,lenky-1)
    tauz=Diagonal([ones(ComplexF64,lenev); -ones(ComplexF64,lenev)])
    ev=bev[:,mid+1:end,:,:]
    for iy in 1:lenkx,ix in 1:lenkx,ii in 1:mid
        tmp=ev[:,ii,ix,iy]
        ev[:,ii,ix,iy]=ev[:,ii,ix,iy]/sqrt(tmp'*tauz*tmp)
    end
    for iy in 1:lenky-1,ix in 1:lenkx-1
        #=
        if ix+1<lenkx
            ixn=ix+1
        else
            ixn=1
        end
        if iy+1<lenky
            iyn=iy+1
        else
            iyn=1
        end
        =#
        ixn=ix+1
        iyn=iy+1
        for ii in 1:mid
            u1=ev[:,ii,ix,iy]'*tauz*ev[:,ii,ixn,iy]
            u2=ev[:,ii,ixn,iy]'*tauz*ev[:,ii,ixn,iyn]
            u3=ev[:,ii,ixn,iyn]'*tauz*ev[:,ii,ix,iyn]
            u4=ev[:,ii,ix,iyn]'*tauz*ev[:,ii,ix,iy]
            bcav[ii,ix,iy]=-angle(u1*u2*u3*u4)
        end
    end

    for ii in 1:mid
        tmp=0.0
        for iy in 1:lenky-1,ix in 1:lenkx-1
            tmp+=bcav[ii,ix,iy]
        end
        Chern[ii]=tmp/2/pi
    end
    return Chern,bcav
end

#####################################################
#             part ben and export
#####################################################
function myexport(ben2d::Array{ComplexF64,3},bcav::Array{Float64,3},
    chern::Array{Float64,1},optn::Int,nb::Int,t::Float64,Gmax::Int,
    nk2d::Int,v0::Float64,m0::Float64,gg::Float64,fg::Float64,ph::String)
    try
        mkdir("ben"*ph)
    catch
        nothing
    end
    for ii in 1:2*nb
        writedlm("ben"*ph*"/ben"*string(ii),real(ben2d[ii,:,:]))
    end
    for ii in 1:nb
        writedlm("ben"*ph*"/bcav"*string(ii),bcav[ii,:,:])
    end
    para=[v0,m0,optn,gg,fg,Gmax]
    paraname="v0,m0,optn,gg,fg,Gmax"
    writedlm("ben"*ph*"/Chern",
    [[string(now()),"time used:",time()-t,paraname,para,"Chern="];chern])
end

include("/Users/zylt/Desktop/soc-jl/DirectlySolve/optin.jl")

function main1d()
    t=time()
    Gmax,nb,optn=5,8,4
    v0,guu,gdd,gud=2.0,0.174868875,0.174868875,0.17446275 #guu=0.175275 gdd=0.17446275
    m0,mz=0.0,0.00035
    nk1d=100
    lenb1,lenb2=sqrt(2),sqrt(2)
    b1,b2=[lenb1,0.0],[0.0,lenb2]

    Gkx,Gky,lenGk=CalGk(b1,b2,Gmax)
    mat=Caloffm(Gkx,Gky,m0,v0,lenGk)

    kx,ky=symline([0 0;lenb1/2 0;lenb1/2 lenb2/2;0 0],nk1d)
    #ben1d=enband1D(mat,Gkx,Gky,lenGk,kx,ky,b1,b2,v0,mz,nb)
    println(time()-t,"s")
    #
    ind,lenind=myind(Gkx,Gky,lenGk)
    println(time()-t,"s index")
    u0,ev0=maincoe(Gkx,Gky,m0,v0,lenGk,b1,b2,optn,ind,lenind,guu,gdd,gud,mz)
    println(time()-t,"s optin")
    ben1d=eigBdgM1D(ind,lenind,mat,lenGk,Gkx,Gky,b1,b2,guu,gdd,gud,v0,nb,kx,ky,u0,ev0,mz)
    println(time()-t,"s end")
    return ben1d
end
ben1d=main1d()
#GR.plot(real(ben1d)')
GR.plot(real(ben1d[9:end,:]'))
#writedlm("ben5010942/ben1d",real(ben1d[9:end,:]))
#writedlm("ben5010942/en1d",real(ben1d))

function main2d()
    t=time()
    Gmax,nb=5,8
    v0,gg,fg=2.0,0.175275,0.995
    optn=4
    m0,mz=0.1,0.0
    nk2d=150
    lenb1=sqrt(2)
    lenb2=sqrt(2)
    b1,b2=[lenb1,0.0],[0.0,lenb2]

    ph=pathout(Gmax,gg,fg,optn,2)
    Gkx,Gky,lenGk=CalGk(b1,b2,Gmax)
    kx,ky=bz2d([-lenb1/2 0;lenb1/2 0],nk2d,0.0,t)

    mat=Caloffm(Gkx,Gky,m0,v0,lenGk,t)
    #ben2d=enband2D(mat,Gkx,Gky,lenGk,kx,ky,b1,b2,v0,mz,nb,t)
    #
    ind,lenind=myind(Gkx,Gky,lenGk,t)
    u0,ev0=maincoe(Gkx,Gky,m0,v0,lenGk,b1,b2,optn,ind,lenind,gg,fg,mz,t)
    ben2d,bev2d=eigBdgM2D(ind,lenind,mat,lenGk,Gkx,Gky,b1,b2,gg,v0,nb,u0,ev0,fg,kx,ky,ph,t)
    #
    Chern,bcav=Bcuvat(bev2d)
    myexport(ben2d,bcav,Chern,optn,nb,t,Gmax,nk2d,v0,m0,gg,fg,ph)
    nothing
    #
    return ben2d
end
#ben2d=main2d()
#GR.heatmap(real(ben2d[4,:,:]))
#surface(ben2d[:,:,:])
