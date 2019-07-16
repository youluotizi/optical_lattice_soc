using LinearAlgebra
using GR

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

function symline(pointlist::Array{Float64,2},num::Int,t::Float64)
    lenlist=size(pointlist,1)
    linbz=Array{Float64,2}(undef,2,(num-1)*(lenlist-1)+lenlist)
    points=pointlist'
    linbz[:,1]=points[:,1]
    kk=1

    for ii in 1:lenlist-1
        vec_tmp=points[:,ii+1]-points[:,ii]
        vec_tmp=vec_tmp./num
        lentmp=norm(vec_tmp)
        for jj in 1:num
            kk+=1
            linbz[:,kk]=points[:,ii]+jj*vec_tmp
        end
        linbz[:,kk]=points[:,ii+1]
    end
    println(time()-t,"s 1D bz completed")
    return linbz[1,:],linbz[2,:]
end
function linsym(plist::Array{Float64,2},num::Int,t::Float64)
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
        p0=p[:,ii];   vc=p[:,ii+1].-p0
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

    println(time()-t,"s 1D bz completed")
    return bz[1,1:kk],bz[2,1:kk],rr
end
function bz2d(pointlist::Array{Float64,2},num::Int,span::Float64,t::Float64)
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

function Caloffm(Gkx::Array{Int,1},Gky::Array{Int,1},m0::Float64,
    v0::Float64,lenGk::Int,t::Float64)
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
    println(time()-t,"s offmat")
    return mat
end

function Caloffmu(Gkx::Array{Int,1},Gky::Array{Int,1},m0::Float64,
    v0::Float64,lenGk::Int,t::Float64)
    mat=zeros(ComplexF64,2*lenGk,2*lenGk)
    for jj in 1:lenGk,ii in 1:lenGk
        t1=Gkx[ii]-Gkx[jj]
        t2=Gky[ii]-Gky[jj]
        if (t1==1&&t2==0)||(t1==-1&&t2==0)||(t1==0&&t2==1)||(t1==0&&t2==-1)
            mat[2*ii-1,2*jj-1]=mat[2*ii,2*jj]=v0/4.0
        end
        if t1==1&&t2==1
            mat[2*ii-1,2*jj]=(1.0-1.0im)*m0/4.0/1.0im
        elseif t1==1&&t2==0
            mat[2*ii-1,2*jj]=(-1.0-1.0im)*m0/4.0/1.0im
        elseif t1==0&&t2==1
            mat[2*ii-1,2*jj]=(1.0+1.0im)*m0/4.0/1.0im
        elseif t1==-1&&t2==-1
            mat[2*ii,2*jj-1]=-(1.0+1.0im)*m0/4.0/1.0im
        elseif t1==-1&&t2==0
            mat[2*ii,2*jj-1]=(1.0-1im)*m0/4.0/1.0im
        elseif t1==0&&t2==-1
            mat[2*ii,2*jj-1]=(-1.0+1im)*m0/4.0/1.0im
        end
    end
    for ii in 1:lenGk
        mat[2*ii-1,2*ii]=(-1.0+1im)*m0/4.0/1im
        mat[2*ii,2*ii-1]=-(-1.0-1im)*m0/4.0/1im
    end
    println(time()-t,"s offmat")
    return mat
end

function enband1D(mat::Array{ComplexF64,2},Gkx::Array{Int,1},Gky::Array{Int,1},
    lenGk::Int,kx::Array{Float64,1},ky::Array{Float64,1},b1::Array{Float64,1},
    b2::Array{Float64,1},v0::Float64,mz::Float64,nb::Int,t::Float64)

    lenkx=length(kx)
    en=Array{Float64,2}(undef,nb,lenkx)
    ev=Array{ComplexF64,3}(undef,2*lenGk,nb,lenkx)

    for ik in 1:lenkx
        for mm in 1:lenGk
            vec_tmp=[kx[ik],ky[ik]]+Gkx[mm]*b1+Gky[mm]*b2
            tmp=vec_tmp'*vec_tmp+v0
            mat[2*mm-1,2*mm-1]=tmp+mz
            mat[2*mm,2*mm]=tmp-mz
        end
        #en_tmp=eigvals(Hermitian(mat))
        en_tmp,ev_tmp=eigen(Hermitian(mat))
        pt=partialsortperm(en_tmp,1:nb)
        en[:,ik]=en_tmp[pt]
        ev[:,:,ik]=ev_tmp[:,pt]
    end
    println(time()-t,"s single particle 1D band")
    return en,ev
end

function enband2D(mat::Array{ComplexF64,2},Gkx::Array{Int,1},Gky::Array{Int,1},
    lenGk::Int,kx::Array{Float64,1},ky::Array{Float64,1},b1::Array{Float64,1},
    b2::Array{Float64,1},v0::Float64,mz::Float64,nb::Int,t::Float64)

    lenky=length(ky)
    lenkx=length(kx)
    Mg=Array{ComplexF64,2}(undef,lenkx,lenky)
    for iky in 1:lenky,ikx in 1:lenkx
        for mm in 1:lenGk
            vec_tmp=[kx[ikx],ky[iky]]+Gkx[mm]*b1+Gky[mm]*b2
            tmp=vec_tmp'*vec_tmp+v0
            mat[2*mm-1,2*mm-1]=tmp+mz
            mat[2*mm,2*mm]=tmp-mz
        end
        en_tmp,ev_tmp=eigen(Hermitian(mat))
        pt=argmin(en_tmp)
        ev=ev_tmp[:,pt]
        Mg[ikx,iky]=mglize(ev)
        #Mg[ikx,iky]=en_tmp[pt]
    end
    println(time()-t,"s single particle 2D band")
    return Mg
end

function enband1Du(mat::Array{ComplexF64,2},Gkx::Array{Int,1},Gky::Array{Int,1},
    lenGk::Int,kx::Array{Float64,1},ky::Array{Float64,1},b1::Array{Float64,1},
    b2::Array{Float64,1},v0::Float64,mz::Float64,nb::Int,t::Float64)

    lenkx=length(kx)
    en=Array{Float64,2}(undef,nb,lenkx)
    ev=Array{ComplexF64,3}(undef,2*lenGk,nb,lenkx)

    for ik in 1:lenkx
        for mm in 1:lenGk
            vec_tmp=[kx[ik],ky[ik]]+Gkx[mm]*b1+Gky[mm]*b2
            tmp=vec_tmp'*vec_tmp+v0
            mat[2*mm-1,2*mm-1]=tmp+mz
            vec_tmp=vec_tmp+[1.0,1.0]
            tmp=vec_tmp'*vec_tmp+v0
            mat[2*mm,2*mm]=tmp-mz
        end
        #en_tmp=eigvals(Hermitian(mat))
        en_tmp,ev_tmp=eigen(Hermitian(mat))
        pt=partialsortperm(en_tmp,1:nb)
        en[:,ik]=en_tmp[pt]
        ev[:,:,ik]=ev_tmp[:,pt]
    end
    println(time()-t,"s single particle 1D band")
    return en,ev
end

function enband2Du(mat::Array{ComplexF64,2},Gkx::Array{Int,1},Gky::Array{Int,1},
    lenGk::Int,kx::Array{Float64,1},ky::Array{Float64,1},b1::Array{Float64,1},
    b2::Array{Float64,1},v0::Float64,mz::Float64,nb::Int,t::Float64)

    lenky=length(ky)
    lenkx=length(kx)
    Mg=Array{ComplexF64,2}(undef,lenkx,lenky)
    for iky in 1:lenky,ikx in 1:lenkx
        for mm in 1:lenGk
            vec_tmp=[kx[ikx],ky[iky]]+Gkx[mm]*b1+Gky[mm]*b2
            tmp=vec_tmp'*vec_tmp+v0
            mat[2*mm-1,2*mm-1]=tmp+mz
            vec_tmp=vec_tmp+[1.0,1.0]
            tmp=vec_tmp'*vec_tmp+v0
            mat[2*mm,2*mm]=tmp-mz
        end
        en_tmp,ev_tmp=eigen(Hermitian(mat))
        pt=argmin(en_tmp)
        ev=ev_tmp[:,pt]
        Mg[ikx,iky]=mglize(ev)
        #Mg[ikx,iky]=en_tmp[pt]
    end
    println(time()-t,"s single particle 2D band")
    return Mg
end

function mglize(ev::Array{ComplexF64,1})
    lenev=Int(length(ev)/2)
    sgz::ComplexF64=0.0im
    for ii in 1:lenev
        sgz+=conj(ev[2*ii-1])*ev[2*ii-1]-conj(ev[2*ii])*ev[2*ii]
    end
    return sgz
end

function main1d()
    t=time()
    Gmax=5
    gg,fg=0.3,1.0
    optn=10
    v0,m0,nb=2.0,0.1,4
    mz=0.001
    nk1d=160
    lenb1=sqrt(2)
    lenb2=sqrt(2)
    b1,b2=[lenb1,0.0],[0.0,lenb2]

    kx,ky,rr=linsym([0 0;lenb1/2 lenb2/2;0 lenb2;0 0],nk1d,t)

    Gkx,Gky,lenGk=CalGk(b1,b2,Gmax)
    mat=Caloffm(Gkx,Gky,m0,v0,lenGk,t)
    ben1d,ev=enband1D(mat,Gkx,Gky,lenGk,kx,ky,b1,b2,v0,mz,nb,t)
    #ben1d=enband2D(mat,Gkx,Gky,lenGk,kx,ky,b1,b2,v0,mz,nb,t);ev=0
    return ben1d,rr,ev
end

function main1du()
    t=time()
    Gmax=5
    gg,fg=0.3,1.0
    optn=10
    v0,m0,nb=4.16,1.32,2
    mz=0.1
    nk1d=160
    lenb1=2.#sqrt(2)
    lenb2=2.#sqrt(2)
    b1,b2=[lenb1,0.0],[0.0,lenb2]

    #kx,ky=symline([0 0;lenb1/2 lenb2/2;0 lenb2/2*sqrt(2);0 0],nk1d,t)
    kx,ky,rr=linsym([0 0;lenb1/2 0;lenb1/2 lenb2/2;0 0],nk1d,t)
    #kx,ky,rr=linsym([0 0;lenb1/4 lenb2/4;0 lenb2/2;0 0],nk1d,t)
    #kx=[0.0,lenb1/2,lenb1/2,0]
    #ky=[0.0,0.0,lenb2/2,0]

    #kx,ky=bz2d([-lenb1/2 -lenb2/2;0 0;lenb1/2 lenb2/2],20,0.0,t)

    Gkx,Gky,lenGk=CalGk(b1,b2,Gmax)
    mat=Caloffmu(Gkx,Gky,m0,v0,lenGk,t)
    ben1d,ev=enband1Du(mat,Gkx,Gky,lenGk,kx,ky,b1,b2,v0,mz,nb,t)
    #ben1d=enband2Du(mat,Gkx,Gky,lenGk,kx,ky,b1,b2,v0,mz,nb,t);ev=0
    return ben1d,rr,ev
end
ben,rr,ev=main1d();
GR.plot(ben')
#GR.heatmap(real(ben))

function sg(ev::Array{ComplexF64,1})
    lenev=Int(length(ev)/2)
    sgz=sgx=sgy=0.0im
    for ii in 1:lenev
        sgx+=ev[2*ii-1]*conj(ev[2*ii])+conj(ev[2*ii-1])*ev[2*ii]
        sgy+=ev[2*ii-1]*conj(ev[2*ii])*1im-1im*conj(ev[2*ii-1])*ev[2*ii]
        sgz+=conj(ev[2*ii-1])*ev[2*ii-1]-conj(ev[2*ii])*ev[2*ii]
    end
    return [sgx,sgy,sgz]
end
