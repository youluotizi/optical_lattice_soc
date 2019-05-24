using Distributed
using DelimitedFiles
using JLD2, FileIO
#using GR #Plots
#nprocs()<5 && addprocs(5-nprocs())
addprocs(40)
@everywhere using LinearAlgebra
@everywhere using SharedArrays


function pathout(Gmax::Int,gg::Float64,fg::Float64,optn::Int)
    tmp=string(gg)
    ph=string(Gmax)*tmp[1]*tmp[3]
    tmp=string(fg)
    ph=ph*tmp[1]*tmp[3]*string(optn)
    return ph
end


function CalGk(Gmax::Int,lenb1::Float64,lenb2::Float64,
    b1::Array{Float64},b2::Array{Float64},t::Float64)
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
    println("lenGk=",kk)
    return Gkx[1:kk],Gky[1:kk],kk
end


####################################################
#               the BZ
####################################################
function discretekx(nkx::Int,lenb1::Float64,t::Float64)
    kx=Float64[0.0]
    ind_kx=Int[0]
    for ii in 1:ceil(Int,nkx/2)
        kx_tmp=ii*lenb1/nkx
        kx_tmp>lenb1/2+1e-6 && break
        push!(kx,kx_tmp)
        push!(ind_kx,ii)
    end
    #
    if (kx[end]+1e-6>lenb1/2)
        kx=[reverse(-kx[2:end-1]);kx]
        ind_kx=[reverse(-ind_kx[2:end-1]);ind_kx]
    else
        kx=[reverse(-kx[2:end]);kx]
        ind_kx=[reverse(-ind_kx[2:end]);ind_kx]
    end
    #=
    kx=[reverse(-kx[2:end]);kx[1:end]]
    ind_kx=[reverse(-ind_kx[2:end]);ind_kx[1:end]]
    =#
    return kx,ind_kx
end
function symline(pointlist::Array{Float64,2},num::Int,t::Float64)
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
    println(time()-t,"s 1D bz completed")
    return linbz[2,:]
end

#pop!(ky)
#ky=[0.0,lenb2/2/3,lenb2/2/3*2,lenb2/2/3*3]

####################################################
#               the Fourier Series
####################################################
function Calfourier(nn::Int,t::Float64)
    if iseven(nn)
        println("Fourier error")
        return nothing
    end
    kk=[ii for ii in -nn:2:nn]
    ck=similar(kk,ComplexF64)
    for ii in 1:length(kk)
        ck[ii]=-2im/(kk[ii]*pi)
    end
    println(time()-t,"s Fourier completed")
    return kk,ck
end



##################################################
#        single particle part of Hamiltonian
#################################################
function Caloffm(Gkx::Array{Int,1},Gky::Array{Int,1},m0::Float64,v0::Float64,
    lenGk::Int,kx::Array{Float64,1},indkx::Array{Int,1},t::Float64)
    lenkx::Int=length(kx)
    mat=zeros(ComplexF64,2*lenGk*lenkx,2*lenGk*lenkx)
    for nx in 1:lenkx,mx in 1:lenkx
        -indkx[mx]+indkx[nx]!=0 && continue
        mm=(mx-1)*2*lenGk; nn=(nx-1)*2*lenGk
        for jj in 1:lenGk,ii in 1:lenGk
            t1=Gkx[ii]-Gkx[jj]
            t2=Gky[ii]-Gky[jj]
            if (t1==1&&t2==-1)||(t1==-1&&t2==1)||(t1==1&&t2==1)||(t1==-1&&t2==-1)
                mat[2*ii-1+mm,2*jj-1+nn]=mat[2*ii+mm,2*jj+nn]=v0/4.0
            end
            if t1==1&&t2==0
                mat[2*ii-1+mm,2*jj+nn]=(1.0-1.0im)*m0/4.0/1.0im
                mat[2*ii+mm,2*jj-1+nn]=(1.0+1.0im)*m0/4.0/1.0im
            elseif t1==-1&&t2==0
                mat[2*ii-1+mm,2*jj+nn]=(-1.0+1im)*m0/4.0/1im
                mat[2*ii+mm,2*jj-1+nn]=(-1.0-1im)*m0/4.0/1im
            elseif t1==0&&t2==1
                mat[2*ii-1+mm,2*jj+nn]=(1.0+1im)*m0/4.0/1im
                mat[2*ii+mm,2*jj-1+nn]=(1.0-1im)*m0/4.0/1im
            elseif t1==0&&t2==-1
                mat[2*ii-1+mm,2*jj+nn]=(-1.0-1im)*m0/4.0/1im
                mat[2*ii+mm,2*jj-1+nn]=(-1.0+1im)*m0/4.0/1im
            end
        end
    end
    println(time()-t,"s offmat ",sum(abs.(mat.-mat')))
    return mat
end


################################################
#             single particle band
################################################
function enband1D(mat::Array{ComplexF64,2},Gkx::Array{Int,1},Gky::Array{Int,1},lenGk::Int,
    kx::Array{Float64,1},ky::Array{Float64,1},b1::Array{Float64,1},b2::Array{Float64,1},
    v0::Float64,nb::Int,t::Float64)

    lenkx=length(kx); lenky=length(ky)
    #en=Array{Float64,2}(undef,nb,lenky)
    en=zeros(Float64,nb,lenky)

    for iy in 1:lenky
        for ix in 1:lenkx
            tmp=(ix-1)*lenGk*2
            for mm in 1:lenGk
            vec_tmp=[kx[ix],ky[iy]]+Gkx[mm]*b1+Gky[mm]*b2
            mat[tmp+2*mm-1,tmp+2*mm-1]=mat[tmp+2*mm,tmp+2*mm]=dot(vec_tmp,vec_tmp)+v0
            end
        end
        en_tmp=eigvals(Hermitian(mat))
        pt=partialsortperm(en_tmp,1:nb)
        en[:,iy]=en_tmp[pt]
    end
    println(time()-t,"s single particle 1D band")
    return en#,ev
end
#en1d=enband1D(mat,Gkx,Gky,lenGk,kx,ky,b1,b2,v0,nb,t)


#########################################
#             the index
#######################################
function myind2ex(Gkx::Array{Int,1},Gky::Array{Int,1},lenGk::Int,t::Float64)
    index0=Array{Int,2}(undef,4,lenGk^3*floor(Int,lenGk/2))
    ind0::Int=1
    @inbounds for ii in 1:lenGk,jj in 1:lenGk,mm in 1:lenGk
        tm::Int=-Gkx[ii]-Gkx[jj]+Gkx[mm]
        tn::Int=-Gky[ii]-Gky[jj]+Gky[mm]
        for nn in 1:lenGk
            if tn+Gky[nn]==0&&tm+Gkx[nn]==0
                index0[:,ind0]=[ii,jj,mm,nn]
                ind0+=1
                break
            end
        end
    end
    ind0-=1
    println(time()-t,"s index=",ind0)
    return index0[:,1:ind0],ind0
end



#########################################
#             ground state
#######################################
function u0f4(kF::Array{Int,1},ckF::Array{ComplexF64,1})
    lenkF=length(kF)
    tmp::ComplexF64=0.0+0.0im
    for ii in 1:lenkF,jj in 1:lenkF,mm in 1:lenkF,nn in 1:lenkF
        if -kF[ii]-kF[jj]+kF[mm]+kF[nn]==0
            tmp+=conj(ckF[ii]*ckF[jj])*ckF[mm]*ckF[nn]
        end
    end
    return tmp
end
function u0f3(kF::Array{Int,1},ckF::Array{ComplexF64,1})
    lenkF=length(kF)
    tmp::ComplexF64=0.0+0.0im
    for ii in 1:lenkF,jj in 1:lenkF,mm in 1:lenkF,nn in 1:lenkF
        if -kF[jj]+kF[mm]+kF[nn]==0
            tmp+=conj(ckF[jj])*ckF[mm]*ckF[nn]
        end
    end
    println("f3",tmp)
    tmp=0.0+0.0im
    for ii in 1:lenkF,jj in 1:lenkF,mm in 1:lenkF,nn in 1:lenkF
        if -kF[jj]-kF[mm]+kF[nn]==0
            tmp+=conj(ckF[jj]*ckF[mm])*ckF[nn]
        end
    end
    println("f3",tmp)
    return tmp
end
function u0f2(ckF::Array{ComplexF64,1})
    tmp::ComplexF64=0.0+0.0im
    for ii in ckF
        tmp+=abs2(ii)
    end
    return tmp
end

function u0en0(ind::Array{Int,2},lenind::Int,lenGk::Int,Gkx::Array{Int,1},Gky::Array{Int,1},
    kF::Array{Int,1},ckF::Array{ComplexF64,1},b1::Array{Float64,1},b2::Array{Float64,1},
    v0::Float64,gg::Float64,fg::Float64,ph::String,t::Float64)
    println("--------- ground state ----------")
    coe_tmp=readdlm(ph*"/xopt"*ph*"1")
    nummin=size(coe_tmp,1)
    println("numin:",size(coe_tmp))

    ev=load(ph*"/ev"*ph*".jld2","ev")
    en=load(ph*"/en"*ph*".jld2","en")
    coe1=Array{ComplexF64,1}(undef,nummin)
    ev1=zeros(ComplexF64,2*lenGk)
    for ii in 1:nummin
        coe1[ii]=complex(coe_tmp[ii,1],coe_tmp[ii,2])
        ev1[:].+=ev[:,ii].*(coe1[ii]/2)
    end
    println("normal:",sum(abs2.(coe1)))
    en0::Float64=0.0
    for ii in 1:nummin
        en0+=abs2(coe1[ii])*en[ii]
    end

    coe_tmp.=readdlm(ph*"/xopt"*ph*"2")
    coe2=similar(coe1)
    ev2=zeros(ComplexF64,2*lenGk)
    for ii in 1:nummin
        coe2[ii]=complex(coe_tmp[ii,1],coe_tmp[ii,2])
        ev2[:].+=ev[:,ii].*(coe2[ii]/2)
    end

    println("orth:",ev1'*ev2)
    ev1tmp=copy(ev1)
    #ev2.=copy(ev1)
    ev1.=ev1tmp.+ev2
    ev2.=ev1tmp.-ev2
    #f4=u0f4(kF,ckF)
    #f2=u0f2(ckF)
    u1::ComplexF64=u2::ComplexF64=u3::ComplexF64=u4::ComplexF64=0.0+0.0im
    u1f::ComplexF64=u2f::ComplexF64=u3f::ComplexF64=u4f::ComplexF64=0.0+0.0im
    @inbounds for ii in 1:lenind
        t1,t2,t3,t4=view(ind,:,ii)
        u1+=conj(ev1[2*t1-1]*ev1[2*t2-1])*ev1[2*t3-1]*ev1[2*t4-1]
        u4+=conj(ev2[2*t1-1]*ev2[2*t2-1])*ev2[2*t3-1]*ev2[2*t4-1]
        u2+=conj(ev1[2*t1-1]*ev1[2*t2-1])*ev2[2*t3-1]*ev2[2*t4-1]
        u3+=conj(ev1[2*t1-1]*ev2[2*t2-1])*ev1[2*t3-1]*ev2[2*t4-1]
        u3+=conj(ev1[2*t1-1]*ev2[2*t2-1])*ev2[2*t3-1]*ev1[2*t4-1]

        u1+=conj(ev1[2*t1]*ev1[2*t2])*ev1[2*t3]*ev1[2*t4]
        u4+=conj(ev2[2*t1]*ev2[2*t2])*ev2[2*t3]*ev2[2*t4]
        u2+=conj(ev1[2*t1]*ev1[2*t2])*ev2[2*t3]*ev2[2*t4]
        u3+=conj(ev1[2*t1]*ev2[2*t2])*ev1[2*t3]*ev2[2*t4]
        u3+=conj(ev1[2*t1]*ev2[2*t2])*ev2[2*t3]*ev1[2*t4]

        u1f+=conj(ev1[2*t1-1]*ev1[2*t2])*ev1[2*t3]*ev1[2*t4-1]
        u4f+=conj(ev2[2*t1-1]*ev2[2*t2])*ev2[2*t3]*ev2[2*t4-1]
        u2f+=conj(ev1[2*t1-1]*ev1[2*t2])*ev2[2*t3]*ev2[2*t4-1]
        u3f+=conj(ev1[2*t1-1]*ev2[2*t2])*ev1[2*t3]*ev2[2*t4-1]
        u3f+=conj(ev1[2*t1-1]*ev2[2*t2])*ev2[2*t3]*ev1[2*t4-1]

        u1f+=conj(ev1[2*t1]*ev1[2*t2-1])*ev1[2*t3-1]*ev1[2*t4]
        u4f+=conj(ev2[2*t1]*ev2[2*t2-1])*ev2[2*t3-1]*ev2[2*t4]
        u2f+=conj(ev1[2*t1]*ev1[2*t2-1])*ev2[2*t3-1]*ev2[2*t4]
        u3f+=conj(ev1[2*t1]*ev2[2*t2-1])*ev1[2*t3-1]*ev2[2*t4]
        u3f+=conj(ev1[2*t1]*ev2[2*t2-1])*ev2[2*t3-1]*ev1[2*t4]
    end
    u1=u1+u4+(real(u2)*2+u3*2)
    u1=u1+(u1f+u4f+(real(u2f)*2+u3f*2))*fg
    println("en0 ",en0)
    println("u0: ",u1*gg+en0)
    println(time()-t,"s u0 completed")
    #ev2.=0.0+0.0im;ev1.=ev1tmp.*2
    return ev1,ev2,real(u1*gg)+en0/2
end

#=
function u0check(ind::Array{Int,2},lenind::Int,mat::Array{ComplexF64,2},lenGk::Int,Gkx::Array{Int,1},
    Gky::Array{Int,1},b1::Array{Float64,1},b2::Array{Float64,1},v0::Float64,gg::Float64,nb::Int,fg::Float64,ph::String)
    for mm in 1:lenGk
        vec_tmp=Gkx[mm]*b1+Gky[mm]*b2
        mat[2*mm-1,2*mm-1]=mat[2*mm,2*mm]=vec_tmp'*vec_tmp+v0
    end
    #println("--------- ground state ----------")
    ev=load(ph*"/ev"*ph*".jld2","ev")
    en=load(ph*"/en"*ph*".jld2","en")
    coe_tmp=readdlm(ph*"/xopt"*ph*"1")
    nummin=size(coe_tmp,1)
    println("numin:",size(coe_tmp))
    coe=[complex(coe_tmp[ii,1],coe_tmp[ii,2]) for ii in 1:nummin]
    println("check normal:",sum(abs2.(coe)))

    ev0=zeros(ComplexF64,2*lenGk)
    for ii in 1:nummin
        ev0[:].+=ev[:,ii].*coe[ii]
    end

    coe_tmp.=readdlm(ph*"/xopt"*ph*"2")
    coe2=similar(coe)
    for ii in 1:nummin
        coe2[ii]=complex(coe_tmp[ii,1],coe_tmp[ii,2])
    end
    ev2=zeros(ComplexF64,2*lenGk)
    for ii in 1:nummin
        ev2[:]+=ev[:,ii].*coe2[ii]
    end
    cc=normalize(rand(2))
    ev0.=ev2.*cc[2].+ev0.*cc[1]

    u0::ComplexF64=0.0+0.0im
    @inbounds for ii in 1:lenind
        t1,t2,t3,t4=view(ind,:,ii)
        u0+=conj(ev0[2*t1-1]*ev0[2*t2-1])*ev0[2*t3-1]*ev0[2*t4-1]
        u0+=conj(ev0[2*t1]*ev0[2*t2])*ev0[2*t3]*ev0[2*t4]
        u0+=conj(ev0[2*t1-1]*ev0[2*t2])*ev0[2*t3]*ev0[2*t4-1]*fg
        u0+=conj(ev0[2*t1]*ev0[2*t2-1])*ev0[2*t3-1]*ev0[2*t4]*fg
    end

    en0::Float64=0.0
    for ii in 1:nummin
        en0+=abs2(coe[ii])*en[ii]
    end
    en2::Float64=0.0
    for ii in 1:nummin
        en2+=abs2(coe2[ii])*en[ii]
    end
    println("en1 ",en0)
    #println("en0 check",ev0'*(mat*ev0))
    println("u1: ",ev0'*(mat*ev0)+u0*gg)
 println("-------------------")
    return real(u0*gg)+en0/2,ev0
end
u1,_=u0check(ind,lenind,mat1,lenGk,Gkx,Gky,b1,b2,v0,gg,nb,fg,ph)
=#

#
function edgm1!(matH::SharedArray{ComplexF64,2},ev1::Array{ComplexF64,1},ev2::Array{ComplexF64,1},
    lenGk::Int,fg::Float64,Gkx::Array{Int,1},Gky::Array{Int,1},indkx::Array{Int,1},nkx::Int,
    kF::Array{Int,1},ckF::Array{ComplexF64,1},t::Float64)

    lenkF::Int=length(kF)
    lenkx::Int=length(indkx)
    lenmat::Int=2*lenGk*lenkx
    @sync @distributed for t1 in Base.OneTo(lenGk)
        @inbounds for t2 in Base.OneTo(lenGk),t3 in Base.OneTo(lenGk)
        ty::Int=-Gky[t1]-Gky[t2]+Gky[t3]
        for t4 in Base.OneTo(lenGk)
            ty+Gky[t4]==0 ? (dG=(-Gkx[t1]-Gkx[t2]+Gkx[t3]+Gkx[t4])*nkx) : continue
            for nx in Base.OneTo(lenkx),mx in Base.OneTo(lenkx)
                p1=p2=true
            for ff in Base.OneTo(lenkF)
            if p1 && -indkx[mx]+indkx[nx]+kF[ff]+dG==0
                mm=(mx-1)*2*lenGk
                nn=(nx-1)*2*lenGk

                tmp=(conj(ev1[2*t2-1])*ev2[2*t3-1]+conj(ev2[2*t2-1])*ev1[2*t3-1])*2
                tmp+=(conj(ev1[2*t2])*ev2[2*t3]+conj(ev2[2*t2])*ev1[2*t3])*fg
                tmp*=ckF[ff]
                matH[2*t1-1+mm,2*t4-1+nn]+=tmp
                matH[2*t4-1+nn+lenmat,2*t1-1+mm+lenmat]+=tmp

                tmp=(conj(ev1[2*t2])*ev2[2*t3-1]+conj(ev2[2*t2])*ev1[2*t3-1])*fg*ckF[ff]
                matH[2*t1-1+mm,2*t4+nn]+=tmp
                matH[2*t4+nn+lenmat,2*t1-1+mm+lenmat]+=tmp

                tmp=(conj(ev1[2*t2-1])*ev2[2*t3]+conj(ev2[2*t2-1])*ev1[2*t3])*fg*ckF[ff]
                matH[2*t1+mm,2*t4-1+nn]+=tmp
                matH[2*t4-1+nn+lenmat,2*t1+mm+lenmat]+=tmp

                tmp=(conj(ev1[2*t2])*ev2[2*t3]+conj(ev2[2*t2])*ev1[2*t3])*2
                tmp+=(conj(ev1[2*t2-1])*ev2[2*t3-1]+conj(ev2[2*t2-1])*ev1[2*t3-1])*fg
                tmp*=ckF[ff]
                matH[2*t1+mm,2*t4+nn]+=tmp
                matH[2*t4+nn+lenmat,2*t1+mm+lenmat]+=tmp
                p1=false
                ############# off diag #####################
            end
            if p2 && -indkx[mx]-indkx[nx]+kF[ff]+dG==0
                mm=(mx-1)*2*lenGk
                nn=(nx-1)*2*lenGk
                #
                tmp=(ev1[2*t3-1]*ev2[2*t4-1]+ev2[2*t3-1]*ev1[2*t4-1])*ckF[ff]
                matH[2*t1-1+mm,2*t2-1+nn+lenmat]+=tmp
                matH[2*t2-1+nn+lenmat,2*t1-1+mm]+=conj(tmp)
                #
                tmp=(ev1[2*t3]*ev2[2*t4-1]+ev2[2*t3]*ev1[2*t4-1])*fg*ckF[ff]
                matH[2*t1-1+mm,2*t2+nn+lenmat]+=tmp
                matH[2*t2+nn+lenmat,2*t1-1+mm]+=conj(tmp)

                tmp=(ev1[2*t3-1]*ev2[2*t4]+ev2[2*t3-1]*ev1[2*t4])*fg*ckF[ff]
                matH[2*t1+mm,2*t2-1+nn+lenmat]+=tmp
                matH[2*t2-1+nn+lenmat,2*t1+mm]+=conj(tmp)
                #
                tmp=(ev1[2*t3]*ev2[2*t4]+ev2[2*t3]*ev1[2*t4])*ckF[ff]
                matH[2*t1+mm,2*t2+nn+lenmat]+=tmp
                matH[2*t2+nn+lenmat,2*t1+mm]+=conj(tmp)
                #
                p2=false
            end
            #p1 || p2 || break
            end
            end
        end
    end
    end
    println(time()-t,"s matH completed")
    nothing
end
#
function edgm2!(matH::SharedArray{ComplexF64,2},ev1::Array{ComplexF64,1},ev2::Array{ComplexF64,1},
    lenGk::Int,mat::Array{ComplexF64,2},ind::Array{Int,2},lenind::Int,gg::Float64,fg::Float64,
    indkx::Array{Int,1},u0::Float64,t::Float64)

    lenkx::Int=length(indkx)
    lenmat::Int=2*lenGk*lenkx

    @sync @inbounds @distributed for ii in Base.OneTo(lenind)
        t1,t2,t3,t4=view(ind,:,ii)
        tmp::ComplexF64=(conj(ev1[2*t2-1])*ev1[2*t3-1]+conj(ev2[2*t2-1])*ev2[2*t3-1])*2
        tmp+=(conj(ev1[2*t2])*ev1[2*t3]+conj(ev2[2*t2])*ev2[2*t3])*fg
        for xx in Base.OneTo(lenkx)
            mm=(xx-1)*2*lenGk
            matH[2*t1-1+mm,2*t4-1+mm]+=tmp
            matH[2*t4-1+mm+lenmat,2*t1-1+mm+lenmat]+=tmp
        end
        tmp=(conj(ev1[2*t2])*ev1[2*t3-1]+conj(ev2[2*t2])*ev2[2*t3-1])*fg
        for xx in Base.OneTo(lenkx)
            mm=(xx-1)*2*lenGk
            matH[2*t1-1+mm,2*t4+mm]+=tmp
            matH[2*t4+mm+lenmat,2*t1-1+mm+lenmat]+=tmp
        end
        tmp=(conj(ev1[2*t2-1])*ev1[2*t3]+conj(ev2[2*t2-1])*ev2[2*t3])*fg
        for xx in Base.OneTo(lenkx)
            mm=(xx-1)*2*lenGk
            matH[2*t1+mm,2*t4-1+mm]+=tmp
            matH[2*t4-1+mm+lenmat,2*t1+mm+lenmat]+=tmp
        end
        tmp=(conj(ev1[2*t2])*ev1[2*t3]+conj(ev2[2*t2])*ev2[2*t3])*2
        tmp+=(conj(ev1[2*t2-1])*ev1[2*t3-1]+conj(ev2[2*t2-1])*ev2[2*t3-1])*fg
        for xx in Base.OneTo(lenkx)
            mm=(xx-1)*2*lenGk
            matH[2*t1+mm,2*t4+mm]+=tmp
            matH[2*t4+mm+lenmat,2*t1+mm+lenmat]+=tmp
        end
        ############# off diag #####################
        tmp=ev1[2*t3-1]*ev1[2*t4-1]+ev2[2*t3-1]*ev2[2*t4-1]
        for xx in Base.OneTo(lenkx)
            mm=(xx-1)*2*lenGk
            matH[2*t1-1+mm,2*t2-1+mm+lenmat]+=tmp
            matH[2*t2-1+mm+lenmat,2*t1-1+mm]+=conj(tmp)
        end
        tmp=(ev1[2*t3]*ev1[2*t4-1]+ev2[2*t3]*ev2[2*t4-1])*fg
        for xx in Base.OneTo(lenkx)
            mm=(xx-1)*2*lenGk
            matH[2*t1-1+mm,2*t2+mm+lenmat]+=tmp
            matH[2*t2+mm+lenmat,2*t1-1+mm]+=conj(tmp)
        end
        tmp=(ev1[2*t3-1]*ev1[2*t4]+ev2[2*t3-1]*ev2[2*t4])*fg
        for xx in Base.OneTo(lenkx)
            mm=(xx-1)*2*lenGk
            matH[2*t1+mm,2*t2-1+mm+lenmat]+=tmp
            matH[2*t2-1+mm+lenmat,2*t1+mm]+=conj(tmp)
        end
        tmp=ev1[2*t3]*ev1[2*t4]+ev2[2*t3]*ev2[2*t4]
        for xx in Base.OneTo(lenkx)
            mm=(xx-1)*2*lenGk
            matH[2*t1+mm,2*t2+mm+lenmat]+=tmp
            matH[2*t2+mm+lenmat,2*t1+mm]+=conj(tmp)
        end
    end
    matH.=matH.*gg
    for ii in 1:lenmat
        mat[ii,ii]=0.0+0.0im
    end
    matH[1:lenmat,1:lenmat].+=mat./2
    matH[lenmat+1:end,lenmat+1:end].+=conj.(mat)./2
    println(time()-t,"s matH1 completed")
    matH.=matH.-Diagonal(fill(u0,2*lenmat))
    nothing
end

@everywhere function BdgM1d(matH::Array{ComplexF64,2},lenGk::Int,b1::Array{Float64,1},
    b2::Array{Float64,1},v0::Float64,Gkx::Array{Int,1},Gky::Array{Int,1},kx::Array{Float64,1},ky::Float64,nb::Int)
    lenkx=length(kx)
    lenmat=2*lenGk*lenkx
    tauz=Diagonal([ones(ComplexF64,lenmat); -ones(ComplexF64,lenmat)])
    for mx in 1:lenkx
        xtmp=(mx-1)*lenGk*2
        for mm in 1:lenGk
            vtmp=[kx[mx],ky].+Gkx[mm].*b1.+Gky[mm].*b2
            vdot=(vtmp'*vtmp+v0)/2
            matH[xtmp+2*mm-1,xtmp+2*mm-1]+=vdot
            matH[xtmp+2*mm,xtmp+2*mm]+=vdot
            vtmp.=[kx[mx],-ky].+Gkx[mm].*b1.+Gky[mm].*b2
            vdot=(vtmp'*vtmp+v0)/2
            matH[xtmp+2*mm-1+lenmat,xtmp+2*mm-1+lenmat]+=vdot
            matH[xtmp+2*mm+lenmat,xtmp+2*mm+lenmat]+=vdot
        end
    end
    #=
    println(sum(abs.(matH-matH')))
    ben=eigvals(Hermitian(matH),1:4)
    println("eig:\n",ben)
    writedlm("eigu0",ben)
    exit()
    =#
    lmul!(tauz,matH)
    ben=eigvals(matH)
    pt=partialsortperm(real(ben),1:lenmat+nb)
    #
    return ben[pt[lenmat+1-nb:end]]
end

function eigBdgM1D(matH::SharedArray{ComplexF64,2},lenGk::Int,b1::Array{Float64,1},
    b2::Array{Float64,1},v0::Float64,Gkx::Array{Int,1},Gky::Array{Int,1},
    kx::Array{Float64,1},ky::Array{Float64,1},nb::Int,t::Float64)
    lenky=length(ky)
    ben=Array{ComplexF64,2}(undef,2*nb,lenky)
    ben_tmp=map(y->BdgM1d(matH[:,:],lenGk,b1,b2,v0,Gkx,Gky,kx,y,nb),ky)
    #return ben_tmp[1]
    for iy in 1:lenky
        ben[:,iy]=ben_tmp[iy]
    end
    println(time()-t,"s 1D Bdg band")
    return ben
end

function mainprogram()
    t=time()
    Gmax=6
    optn=90
    gg,fg=0.2,1.0
    nkx=4 # even number
    nky=80
    v0,m0=4.0,3.0
    nb=12*nkx
    lenb1=sqrt(2)
    lenb2=sqrt(2)
    b1,b2=[lenb1,0.0],[0.0,lenb2]

    ph=pathout(Gmax,gg,fg,optn)
    Gkx,Gky,lenGk=CalGk(Gmax,lenb1,lenb2,b1,b2,t)
    kx,indkx=discretekx(nkx,lenb1,t)
    println("kx",kx)
    ky=symline([0 -lenb2/2;0 lenb2/2],nky,t)
    ky1=[ky[2*ii-1] for ii in 1:41]
    #
    println(ky1)
    kF,ckF=Calfourier(301,t)
    mat=Caloffm(Gkx,Gky,m0,v0,lenGk,kx,indkx,t)
    #
    ind,lenind=myind2ex(Gkx,Gky,lenGk,t)
    ev1,ev2,u0=u0en0(ind,lenind,lenGk,Gkx,Gky,kF,ckF,b1,b2,v0,gg,fg,ph,t)
    u0=u0-0.02652872
    lenmat=2*lenGk*length(kx)
    matH=SharedArray{ComplexF64,2}(2*lenmat,2*lenmat)
    matH.=0.0+0.0im
    edgm1!(matH,ev1,ev2,lenGk,fg,Gkx,Gky,indkx,nkx,kF,ckF,t)
    edgm2!(matH,ev1,ev2,lenGk,mat,ind,lenind,gg,fg,indkx,u0,t)
    np=workers()
    rmprocs(np)
    println(typeof(matH))
    cc=1
    ben1d=eigBdgM1D(matH,lenGk,b1,b2,v0,Gkx,Gky,kx,ky1[cc*8-6:cc*8+1],nb,t)
    #=
    ky=symline([0 0;0 lenb2/2],6,t)
    println(ky)
    ben1d=eigBdgM1D(matH,lenGk,b1,b2,v0,Gkx,Gky,kx,ky,nb,t)
    =#
    writedlm("2re"*string(cc),real(ben1d))
    writedlm("2im"*string(cc),imag(ben1d))
    println(time()-t)
    return ben1d
    #
end
ben1d=mainprogram()
#GR.plot(real(ben1d)')
#=
lm=1000
cc=Hermitian(rand(ComplexF64,4*lm,4*lm))
tauz=Diagonal([ones(lm*2);-ones(lm*2)])
cc=tauz*cc
=#
