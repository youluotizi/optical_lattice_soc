using Distributed
using DelimitedFiles
using JLD2,FileIO
using GR 
nprocs()<2 && addprocs(3)
#addprocs(40)
@everywhere using LinearAlgebra
@everywhere using SharedArrays


function pathout(Gmax::Int,v0::Float64,m0::Float64,mz::Float64,optn::Int,pband::Int)
    ph=string(v0)*"-"*string(m0)*"-"*string(mz)*"-"*string(optn)*"-"*string(pband)
    tmp=string(Gmax)*"-"
    for ii in 1:length(ph)
        if ph[ii]=='.'
            continue
        end
        tmp=tmp*ph[ii]
    end
    try
        mkdir(tmp)
    catch
        nothing
    end
    return tmp
end

function CalGk(Gmax::Int,b1::Array{Float64},b2::Array{Float64})
    Gkx=Array{Int,1}(undef,(2*Gmax)^2)
    Gky=Array{Int,1}(undef,(2*Gmax)^2)
    kk::Int=0
    lenb=max(norm(b1),norm(b2))
    pp=Gmax*lenb+1e-6
    @inbounds for jj in -Gmax:Gmax,ii in -Gmax:Gmax
        if norm(ii.*b1.+jj.*b2)<pp
            kk+=1
            Gkx[kk]=ii
            Gky[kk]=jj
        end
    end
    println("lenGk=",kk)
    return Gkx[1:kk],Gky[1:kk],kk
end

####################################################
#               the BZ
####################################################
function discretekx(nkx::Int,lenb1::Float64)
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
    #
    return kx,ind_kx
end

####################################################
#               the Fourier Series
####################################################
function fouriersin(nn::Int)
    if iseven(nn)
        println("Fouriersin error")
        return nothing
    end
    kk=collect(-nn:2:nn)
    ck=similar(kk,ComplexF64)
    for ii in 1:length(kk)
        ck[ii]=-2.0im/(kk[ii]*pi)
    end
    return kk,ck
end

##################################################
#        single particle part of Hamiltonian
#################################################
function Caloffm(Gkx::Array{Int,1},Gky::Array{Int,1},m0::Float64,v0::Float64,
    lenGk::Int,indkx::Array{Int,1})

    lenkx::Int=length(indkx)
    mat=zeros(ComplexF64,2*lenGk*lenkx,2*lenGk*lenkx)
    @inbounds for nx in 1:lenkx,mx in 1:lenkx
        indkx[nx]-indkx[mx]!=0 && continue
        mm=(mx-1)*2*lenGk
        nn=(mx-1)*2*lenGk
        for jj in 1:lenGk,ii in 1:lenGk
            t1=Gkx[ii]-Gkx[jj]
            t2=Gky[ii]-Gky[jj]
            if (t1==1&&t2==-1)||(t1==-1&&t2==1)||(t1==1&&t2==1)||(t1==-1&&t2==-1)
                mat[ii+mm,jj+nn]=mat[ii+lenGk+mm,jj+lenGk+nn] = v0/4.0
            end
            if t1==1 && t2==0
                mat[ii+mm,jj+lenGk+nn] = (-1.0-1.0im)*m0/4.0
                mat[ii+lenGk+mm,jj+nn] = (1.0-1.0im)*m0/4.0
            elseif t1==-1 && t2==0
                mat[ii+mm,jj+lenGk+nn] = (1.0+1.0im)*m0/4.0
                mat[ii+lenGk+mm,jj+nn] = (-1.0+1.0im)*m0/4.0
            elseif t1==0 && t2==1
                mat[ii+mm,jj+lenGk+nn] = (-1.0+1.0im)*m0/4.0
                mat[ii+lenGk+mm,jj+nn] = (1.0+1.0im)*m0/4.0
            elseif t1==0 && t2==-1
                mat[ii+mm,jj+lenGk+nn] = (1.0-1.0im)*m0/4.0
                mat[ii+lenGk+mm,jj+nn] = (-1.0-1.0im)*m0/4.0
            end
        end
    end
    return mat
end


#########################################
#             the index
#######################################
function myind(Gkx::Array{Int,1},Gky::Array{Int,1},lenGk::Int)
    idx=Array{Int,2}(undef,4,lenGk^3*floor(Int,lenGk/2))
    kk::Int=0
    @inbounds for ii in 1:lenGk,jj in 1:lenGk,mm in 1:lenGk
        tx=-Gkx[ii]-Gkx[jj]+Gkx[mm]
        ty=-Gky[ii]-Gky[jj]+Gky[mm]
        for nn in 1:lenGk
            if ty+Gky[nn]==0 && tx+Gkx[nn]==0
                kk+=1
                idx[:,kk].=[ii,jj,mm,nn]
                break
            end
        end
    end
    return idx[:,1:kk],kk
end

#########################################
#             ground state
#######################################
function u0en0(ph::String)
    #
    ev=load(ph*"/ev0.jld2","ev0")::Array{ComplexF64,2}
    lenev=size(ev,1)

    ev1=Array{ComplexF64,1}(undef,lenev)
    ev2=Array{ComplexF64,1}(undef,lenev)

    ev1 .= (ev[:,1].+ev[:,2])./2
    ev2 .= (ev[:,1].-ev[:,2])./2

    println("ev1*ev2\n",ev1'*ev2)
    println("ev2*ev2\n",ev2'*ev2)
    println("ev1*ev1\n",ev1'*ev1)
 #=
    lenkF=length(kF)
    en1::ComplexF64=0.0im
    for (ff,cf) in zip(kF,ckF)
        kx=ff/nkx
        mod(ff,nkx) !=0 && continue
        dG=fld(ff,nkx)*sign(ff)
        tmp=0.0im
        for i1 in 1:lenGk,i2 in 1:lenGk
            Gky[i1]!=Gky[i2] && continue
            if Gkx[i1]==Gkx[i2]+dG
                tmp+=conj(ev1[i1])*ev2[i2]
            end
        end
        en0+=tmp*cf    
    end
    en1=en0/2+2*real(en0)*en0/4
    
    for (ff,cf) in zip(kF,ckF)
        kx=ff/nkx
        for mm in 1:lenGk
            evtmp=[kx,0.0].+ Gkx[mm].*b1 .+ Gky[mm].*b2
            tmp=evtmp'*evtmp+v0
            mat[mm,mm]=mat[mm+lenGk,+lenGk]=tmp
        end
        en1+=(ev2'*mat*ev2)*abs2(cf)
    end

    uu1::ComplexF64=ud1::ComplexF64=dd1::ComplexF64=0.0im
    uu2::ComplexF64=ud2::ComplexF64=dd2::ComplexF64=0.0im
    @inbounds for ii in 1:lenind
        t1,t2,t3,t4=view(ind,:,ii)
        uu1+=conj(ev1[t1]*ev1[t2]+ev2[t1]*ev2[t2])*(ev1[t3]*ev1[t4]+ev2[t3]*ev2[t4])
        uu2+=conj(ev1[t1]*ev2[t2])*ev1[t3]*ev2[t4]

        ud1+=conj(ev1[t1]*ev1[t2+lenGk]+ev2[t1]*ev2[t2+lenGk])*(ev1[t3+lenGk]*ev1[t4]+ev2[t3+lenGk]*ev2[t4])
        ud1+=conj(ev1[t1+lenGk]*ev1[t2]+ev2[t1+lenGk]*ev2[t2])*(ev1[t3]*ev1[t4+lenGk]+ev2[t3]*ev2[t4+lenGk])
        ud2+=conj(ev1[t1]*ev2[t2+lenGk])*ev1[t3+lenGk]*ev2[t4]
        ud2+=conj(ev1[t1+lenGk]*ev2[t2])*ev1[t3]*ev2[t4+lenGk]

        dd1+=conj(ev1[t1+lenGk]*ev1[t2+lenGk]+ev2[t1+lenGk]*ev2[t2+lenGk])*(ev1[t3+lenGk]*ev1[t4+lenGk]+ev2[t3+lenGk]*ev2[t4+lenGk])
        dd2+=conj(ev1[t1+lenGk]*ev2[t2+lenGk])*ev1[t3+lenGk]*ev2[t4+lenGk]
    end
    uu=(uu1+4*uu2)*guu+(ud1+4*ud2)*gud+(dd1+4*ud2)*gdd

    println("u0:  ",uu+en1/2)
 =#
    return ev1,ev2#,0.0#real(uu+en1/2)
end

################################################
#          Bdg matrix with f(x) couple term
################################################
@everywhere function funcmA(kF::Array{Int,1},ckF::Array{ComplexF64,1},
    ev1::Array{ComplexF64,1},ev2::Array{ComplexF64,1},lenkF::Int,tx::Int,
    tm::Int,ty::Int,nkx::Int,lenGk::Int,Gkx::Array{Int,1},Gky::Array{Int,1})

    uu::ComplexF64=dd::ComplexF64=ud::ComplexF64=du::ComplexF64=0.0im
    @inbounds for t2 in 1:lenGk,t3 in 1:lenGk
        ty-Gky[t2]+Gky[t3]==0 ? (dG::Int=(tx-Gkx[t2]+Gkx[t3])*nkx) : continue
        for ff in 1:lenkF
            if tm+kF[ff]+dG==0
                uu+=(conj(ev1[t2])*ev2[t3]+conj(ev2[t2])*ev1[t3])*ckF[ff]
                dd+=(conj(ev1[t2+lenGk])*ev2[t3+lenGk]+conj(ev2[t2+lenGk])*ev1[t3+lenGk])*ckF[ff]

                ud+=(conj(ev1[t2+lenGk])*ev2[t3]+conj(ev2[t2+lenGk])*ev1[t3])*ckF[ff]
                du+=(conj(ev1[t2])*ev2[t3+lenGk]+conj(ev2[t2])*ev1[t3+lenGk])*ckF[ff]
                break
            end
        end
    end
    return uu,dd,ud,du
end

function matAf!(matA::SharedArray{ComplexF64,2},ev1::Array{ComplexF64,1},
    ev2::Array{ComplexF64,1},lenGk::Int,guu::Float64,gud::Float64,gdd::Float64,Gkx::Array{Int,1},
    Gky::Array{Int,1},indkx::Array{Int,1},nkx::Int,kF::Array{Int,1},ckF::Array{ComplexF64,1})

    lenkF::Int=length(kF)
    lenkx::Int=length(indkx)

    @sync @distributed for t1 in 1:lenGk
        @inbounds for t4 in 1:lenGk,nx in 1:lenkx,mx in 1:lenkx

            mm::Int=(mx-1)*2*lenGk
            nn::Int=(nx-1)*2*lenGk
            ty::Int=Gky[t4]-Gky[t1]
            tx::Int=Gkx[t4]-Gkx[t1]
            tm::Int=indkx[nx]-indkx[mx]

            uu,dd,ud,du=funcmA(kF,ckF,ev1,ev2,lenkF,tx,tm,ty,nkx,lenGk,Gkx,Gky)

            matA[t1+mm,t4+nn]=2*guu*uu+gud*dd
            matA[t1+mm,t4+lenGk+nn]=ud*gud
            matA[t1+lenGk+mm,t4+nn]=du*gud
            matA[t1+lenGk+mm,t4+lenGk+nn]=2*gdd*dd+uu*gud
        end
    end
    nothing
end

@everywhere function funcmB(kF::Array{Int,1},ckF::Array{ComplexF64,1},ev1::Array{ComplexF64,1},
    ev2::Array{ComplexF64,1},lenkF::Int,tx::Int,tm::Int,ty::Int,nkx::Int,
    lenGk::Int,Gkx::Array{Int,1},Gky::Array{Int,1})

    uu::ComplexF64=dd::ComplexF64=ud::ComplexF64=du::ComplexF64=0.0im
    @inbounds for t3 in 1:lenGk,t4 in 1:lenGk
        ty+Gky[t3]+Gky[t4]==0 ? (dG::Int=(tx+Gkx[t3]+Gkx[t4])*nkx) : continue
        for ff in 1:lenkF
            if tm+kF[ff]+dG==0
                uu+=(ev1[t3]*ev2[t4]+ev2[t3]*ev1[t4])*ckF[ff]
                dd+=(ev1[t3+lenGk]*ev2[t4+lenGk]+ev2[t3+lenGk]*ev1[t4+lenGk])*ckF[ff]

                ud+=(ev1[t3+lenGk]*ev2[t4]+ev2[t3+lenGk]*ev1[t4])*ckF[ff]
                du+=(ev1[t3]*ev2[t4+lenGk]+ev2[t3]*ev1[t4+lenGk])*ckF[ff]
                break
            end
        end
    end
    return uu,dd,ud,du
end

function matBf!(matB::SharedArray{ComplexF64,2},ev1::Array{ComplexF64,1},
    ev2::Array{ComplexF64,1},lenGk::Int,guu::Float64,gud::Float64,gdd::Float64,Gkx::Array{Int,1},
    Gky::Array{Int,1},indkx::Array{Int,1},nkx::Int,kF::Array{Int,1},ckF::Array{ComplexF64,1})

    lenkF::Int=length(kF)
    lenkx::Int=length(indkx)

    @sync @distributed for t1 in 1:lenGk
        @inbounds for t2 in 1:lenGk,nx in 1:lenkx,mx in 1:lenkx

            mm::Int=(mx-1)*2*lenGk
            nn::Int=(nx-1)*2*lenGk
            ty::Int=-Gky[t1]-Gky[t2]
            tx::Int=-Gkx[t1]-Gkx[t2]
            tm::Int=-indkx[mx]-indkx[nx]

            uu,dd,ud,du=funcmB(kF,ckF,ev1,ev2,lenkF,tx,tm,ty,nkx,lenGk,Gkx,Gky)

            matB[t1+mm,t2+nn]=guu*uu
            matB[t1+mm,t2+lenGk+nn]=ud*gud
            matB[t1+lenGk+mm,t2+nn]=du*gud
            matB[t1+lenGk+mm,t2+lenGk+nn]=gdd*dd
        end
    end
    nothing
end


################################################
#         Bdg matrix without f(x) couple term
################################################
function matAnof!(matA::SharedArray{ComplexF64,2},ev1::Array{ComplexF64,1},
    ev2::Array{ComplexF64,1},lenGk::Int,ind::Array{Int,2},lenind::Int,guu::Float64,
    gud::Float64,gdd::Float64,indkx::Array{Int,1})

    lenkx::Int=length(indkx)
    @sync @inbounds @distributed for ii in 1:lenind
        t1,t2,t3,t4=view(ind,:,ii)
        for xx in 1:lenkx
            mm=(xx-1)*2*lenGk

            tmp::ComplexF64=(conj(ev1[t2])*ev1[t3]+conj(ev2[t2])*ev2[t3])*2*guu
            tmp+=(conj(ev1[t2+lenGk])*ev1[t3+lenGk]+conj(ev2[t2+lenGk])*ev2[t3+lenGk])*gud
            matA[t1+mm,t4+mm]+=tmp

            tmp=(conj(ev1[t2+lenGk])*ev1[t3]+conj(ev2[t2+lenGk])*ev2[t3])*gud
            matA[t1+mm,t4+lenGk+mm]+=tmp
       
            tmp=(conj(ev1[t2])*ev1[t3+lenGk]+conj(ev2[t2])*ev2[t3+lenGk])*gud
            matA[t1+lenGk+mm,t4+mm]+=tmp

            tmp=(conj(ev1[t2+lenGk])*ev1[t3+lenGk]+conj(ev2[t2+lenGk])*ev2[t3+lenGk])*gdd*2
            tmp+=(conj(ev1[t2])*ev1[t3]+conj(ev2[t2])*ev2[t3])*gud
            matA[t1+lenGk+mm,t4+lenGk+mm]+=tmp
        end
    end
    nothing
end

function matBnof!(matB::SharedArray{ComplexF64,2},ev1::Array{ComplexF64,1},
    ev2::Array{ComplexF64,1},lenGk::Int,ind::Array{Int,2},lenind::Int,guu::Float64,
    gud::Float64,gdd::Float64,indkx::Array{Int,1})

    lenkx::Int=length(indkx)

    lenMb::Int=0
    Mb=Array{Int,2}(undef,2,2*lenkx)
    @inbounds for mx in 1:lenkx,nx in 1:lenkx
        if indkx[mx]+indkx[nx]==0
            lenMb+=1
            Mb[1,lenMb]=mx
            Mb[2,lenMb]=nx
        end
    end

    @sync @inbounds @distributed for ii in 1:lenind
        t1,t2,t3,t4=view(ind,:,ii)
        for xx in 1:lenMb
            mm=(Mb[1,xx]-1)*2*lenGk
            nn=(Mb[2,xx]-1)*2*lenGk 

            tmp::ComplexF64=(ev1[t3]*ev1[t4]+ev2[t3]*ev2[t4])*guu
            matB[t1+mm,t2+nn]+=tmp

            tmp=(ev1[t3+lenGk]*ev1[t4]+ev2[t3+lenGk]*ev2[t4])*gud
            matB[t1+mm,t2+lenGk+nn]+=tmp

            tmp=(ev1[t3]*ev1[t4+lenGk]+ev2[t3]*ev2[t4+lenGk])*gud
            matB[t1+lenGk+mm,t2+nn]+=tmp

            tmp=(ev1[t3+lenGk]*ev1[t4+lenGk]+ev2[t3+lenGk]*ev2[t4+lenGk])*gdd
            matB[t1+lenGk+mm,t2+lenGk+nn]+=tmp
        end
    end
    nothing
end

function matBnof2!(matB::SharedArray{ComplexF64,2},Gkx::Array{Int,1},Gky::Array{Int,1},
    lenGk::Int,indkx::Array{Int,1},nkx::Int,ev1::Array{ComplexF64,1},ev2::Array{ComplexF64,1},
    guu::Float64,gud::Float64,gdd::Float64)

    lenkx::Int=length(indkx)

    lenMb::Int=0
    Mb=Array{Int,2}(undef,2,2)
    for mx in 1:lenkx,nx in 1:lenkx
        if -indkx[mx]-indkx[nx]==-nkx
            lenMb+=1
            Mb[1,lenMb]=mx
            Mb[2,lenMb]=nx
        end
    end
    
    if lenMb==0
        println("kx no couple with Gkx")
        return nothing
    end

    @inbounds for t1 in 1:lenGk,t2 in 1:lenGk,t3 in 1:lenGk
        tx::Int=-Gkx[t1]-Gkx[t2]+Gkx[t3]
        ty::Int=-Gky[t1]-Gky[t2]+Gky[t3]
        for t4 in 1:lenGk
            if ty+Gky[t4]==0 && tx+Gkx[t4]==1
                for xx in 1:lenMb
                    mm=(Mb[1,xx]-1)*2*lenGk
                    nn=(Mb[2,xx]-1)*2*lenGk

                    tmp::ComplexF64=(ev1[t3]*ev1[t4]+ev2[t3]*ev2[t4])*guu
                    matB[t1+mm,t2+nn]+=tmp
        
                    tmp=(ev1[t3+lenGk]*ev1[t4]+ev2[t3+lenGk]*ev2[t4])*gud
                    matB[t1+mm,t2+lenGk+nn]+=tmp
        
                    tmp=(ev1[t3]*ev1[t4+lenGk]+ev2[t3]*ev2[t4+lenGk])*gud
                    matB[t1+lenGk+mm,t2+nn]+=tmp
        
                    tmp=(ev1[t3+lenGk]*ev1[t4+lenGk]+ev2[t3+lenGk]*ev2[t4+lenGk])*gdd
                    matB[t1+lenGk+mm,t2+lenGk+nn]+=tmp
                end
            end
        end
    end
    nothing
end

function edgmH(lenGk::Int,guu::Float64,gud::Float64,gdd::Float64,Gkx::Array{Int,1},
    Gky::Array{Int,1},indkx::Array{Int,1},nkx::Int,kF::Array{Int,1},ckF::Array{ComplexF64,1},
    ev1::Array{ComplexF64,1},ev2::Array{ComplexF64,1},ind::Array{Int,2},lenind::Int,tt::Float64)

    lenmat=2*lenGk*nkx
    matA=SharedArray{ComplexF64,2}(lenmat,lenmat)
    matB=SharedArray{ComplexF64,2}(lenmat,lenmat)
    matA.=0.0+0.0im
    matB.=0.0+0.0im

    matAf!(matA,ev1,ev2,lenGk,guu,gud,gdd,Gkx,Gky,indkx,nkx,kF,ckF)
    println(time()-tt,"s matAf")
    matBf!(matB,ev1,ev2,lenGk,guu,gud,gdd,Gkx,Gky,indkx,nkx,kF,ckF)
    println(time()-tt,"s matBf")

    matAnof!(matA,ev1,ev2,lenGk,ind,lenind,guu,gud,gdd,indkx)
    matBnof!(matB,ev1,ev2,lenGk,ind,lenind,guu,gud,gdd,indkx)
    println(time()-tt,"s matnof")
    
    matBnof2!(matB,Gkx,Gky,lenGk,indkx,nkx,ev1,ev2,guu,gud,gdd)
    println(time()-tt,"s mat_couple")

    println("matA:",sum(abs.(matA' .-matA)))
    println("matB:",sum(abs.(matB' .-matB)))
    println("matB:",sum(abs.(conj.(matB') .-matB)))

    matH=Array{ComplexF64,2}(undef,2*lenmat,2*lenmat)
    matH[1:lenmat,1:lenmat].=copy(matA)
    matH[lenmat+1:end,1:lenmat].=matB'
    matH[1:lenmat,lenmat+1:end].=copy(matB)
    matH[lenmat+1:end,lenmat+1:end].=conj.(matA)
    println(time()-tt,"s Hend")
    return matH
end

#########################################
#             chemical potential
#########################################
function funcmu!(matH::Array{ComplexF64,2},lenkx::Int,lenGk::Int,kx::Array{Float64,1},
    Gkx::Array{Int,1},Gky::Array{Int,1},b1::Array{Float64,1},b2::Array{Float64,1},
    v0::Float64,pband::Int)

    matH1=copy(matH)
    lenmat=2*lenGk*lenkx
    vtmp=Array{Float64,1}(undef,2)

    for mx in 1:lenkx
        xtmp=(mx-1)*lenGk*2
        @inbounds for mm in 1:lenGk
            vtmp.=[kx[mx],0.0].+Gkx[mm].*b1.+Gky[mm].*b2
            vdot=(dot(vtmp,vtmp)+v0)/2

            matH1[xtmp+mm,xtmp+mm]+=vdot
            matH1[xtmp+mm+lenGk,xtmp+mm+lenGk]+=vdot

            matH1[xtmp+mm+lenmat,xtmp+mm+lenmat]+=vdot
            matH1[xtmp+mm+lenGk+lenmat,xtmp+mm+lenGk+lenmat]+=vdot
        end
    end

    u0=eigvals(Hermitian(matH1),1:2*pband*lenkx+2)
    utmp=u0[end-1]
    println(sum(abs.(matH1'-matH1))," Δu=",utmp)
    for ii in 1:2*lenmat
        matH[ii,ii]-=utmp
    end
    tauz=Diagonal([ones(ComplexF64,lenmat); -ones(ComplexF64,lenmat)])
    lmul!(tauz,matH)
end

function matHmu!(matH::Array{ComplexF64,2},mat::Array{ComplexF64,2},lenGk::Int,
    Gkx::Array{Int,1},Gky::Array{Int,1},b1::Array{Float64,1},b2::Array{Float64,1},
    kx::Array{Float64,1},v0::Float64,nkx::Int,ph::String,pband::Int)

    lenkx=length(kx)
    lenmat=2*lenGk*lenkx
    for ii in 1:lenmat
        mat[ii,ii]=0.0im
    end
    matH[1:lenmat,1:lenmat].+=mat./2
    matH[lenmat+1:end,lenmat+1:end].+=conj.(mat)./2

    funcmu!(matH,lenkx,lenGk,kx,Gkx,Gky,b1,b2,v0,pband)
    save(ph*"/matH"*string(nkx)*".jld2","matA",matH[1:lenmat,1:lenmat],"matB",matH[1:lenmat,lenmat+1:end])
    nothing
end

function BdgM1d(matH::Array{ComplexF64,2},lenGk::Int,b1::Array{Float64,1},b2::Array{Float64,1},
    v0::Float64,Gkx::Array{Int,1},Gky::Array{Int,1},kx::Array{Float64,1},ky::Float64,nb::Int)
    lenkx=length(kx)
    lenmat=2*lenGk*lenkx
   
    vtmp=Array{Float64,1}(undef,2)
    @inbounds for mx in 1:lenkx
        xtmp=(mx-1)*lenGk*2
        for mm in 1:lenGk
            vtmp.=[kx[mx],ky].+Gkx[mm].*b1.+Gky[mm].*b2
            vdot::Float64 = (dot(vtmp,vtmp)+v0)/2
            matH[xtmp+mm,xtmp+mm]+=vdot
            matH[xtmp+mm+lenGk,xtmp+mm+lenGk]+=vdot

            vtmp.=[kx[mx],-ky].+Gkx[mm].*b1.+Gky[mm].*b2
            vdot=(dot(vtmp,vtmp)+v0)/2
            matH[xtmp+mm+lenmat,xtmp+mm+lenmat]-=vdot
            matH[xtmp+mm+lenGk+lenmat,xtmp+mm+lenGk+lenmat]-=vdot
        end
    end

    ben=eigvals(matH)::Array{ComplexF64,1}
    pt=partialsortperm(real(ben),1:lenmat+nb)
    return ben[pt[lenmat+1:end]]
end

function eigBdgM1D(matH::Array{ComplexF64,2},lenGk::Int,b1::Array{Float64,1},
    b2::Array{Float64,1},v0::Float64,Gkx::Array{Int,1},Gky::Array{Int,1},
    kx::Array{Float64,1},ky::Array{Float64,1},nb::Int)

    lenky=length(ky)
    ben=Array{ComplexF64,2}(undef,nb,lenky)

    for iy in 1:lenky
        ben[:,iy].=BdgM1d(matH[:,:],lenGk,b1,b2,v0,Gkx,Gky,kx,ky[iy],nb)
    end
    println("ben_check",sum(abs.(imag.(ben))))
    return real.(ben)
end

function mainmatrix()
    println("------------matrix----------")
    t=time()
    Gmax,nb,optn=5,12,2
    pband=4
    v0,m0,mz=5.2,3.0,0.0
    guu,gdd,gud=0.174868875,0.174868875,0.17446275 #guu=0.175275 gdd=0.17446275
    lenb1,lenb2=sqrt(2),sqrt(2)
    b1,b2=[lenb1,0.0],[0.0,lenb2]

    nkx=6 # even number
    nb=8*nkx

    b1,b2=[sqrt(2),0.0],[0.0,sqrt(2)]

    ph=pathout(Gmax,v0,m0,mz,optn,pband)
    Gkx,Gky,lenGk=CalGk(Gmax,b1,b2)

    kx,indkx=discretekx(nkx,norm(b1))

    kF,ckF=fouriersin(101)
    println(length(kF))
    
    mat=Caloffm(Gkx,Gky,m0,v0,lenGk,indkx)
    println(sum(abs.(mat'-mat)))
    
    ind,lenind=myind(Gkx,Gky,lenGk)
    ev1,ev2=u0en0(ph)

    matH=edgmH(lenGk,guu,gud,gdd,Gkx,Gky,indkx,nkx,kF,ckF,ev1,ev2,ind,lenind,t)
    #rmprocs(nworkers())
    matHmu!(matH,mat,lenGk,Gkx,Gky,b1,b2,kx,v0,nkx,ph,pband)
    println(time()-t,"s matH4")

    return nothing#matH

    ky=collect(0.0:sqrt(2.0)/2/40:sqrt(2.0)/2)
    cc=7
    ky1=ky[6*cc-5:6*cc-1]

    ben1d=eigBdgM1D(matH,lenGk,b1,b2,v0,Gkx,Gky,kx,ky[1:4:end],nb)
    #writedlm(ph*"/ben"*string(nkx),ben1d)
    println(time()-t)
    return ben1d
end
#ben1d=mainmatrix()

function matHload!(matH::Array{ComplexF64,2},ph::String,nkx::Int,lenGk::Int)
    tmpA=load(ph*"/matH"*string(nkx)*".jld2","matA")
    tmpB=load(ph*"/matH"*string(nkx)*".jld2","matB")
    lenmat=size(tmpA,1)
    println(lenmat)
    matH[1:lenmat,1:lenmat].=copy(tmpA)
    matH[lenmat+1:end,1:lenmat].=(tmpB').*(-1.0)
    matH[1:lenmat,lenmat+1:end].=copy(tmpB)
    matH[lenmat+1:end,lenmat+1:end].=conj.(tmpA)*(-1.0)
    nothing
end
function maineig()
    println("------------Fpw-eig-----------")
    t=time()
    Gmax,nb,optn=5,12,2
    pband=4

    v0,m0,mz=5.2,3.0,0.0
    guu,gdd,gud=0.174868875,0.174868875,0.17446275 #guu=0.175275 gdd=0.17446275
    lenb1,lenb2=sqrt(2),sqrt(2)
    b1,b2=[lenb1,0.0],[0.0,lenb2]

    nkx=6 # even number
    nb=8*nkx

    b1,b2=[sqrt(2),0.0],[0.0,sqrt(2)]

    ph=pathout(Gmax,v0,m0,mz,optn,pband)
    Gkx,Gky,lenGk=CalGk(Gmax,b1,b2)

    kx,indkx=discretekx(nkx,norm(b1))
    #
    matH=Array{ComplexF64,2}(undef,4*lenGk*length(kx),4*lenGk*length(kx))
    matHload!(matH,ph,nkx,lenGk)

    ky=collect(0.0:sqrt(2.0)/2/40:sqrt(2.0)/2)
    cc=1
    ky1=ky[6*cc-5:6*cc-1]

    ben1d=eigBdgM1D(matH,lenGk,b1,b2,v0,Gkx,Gky,kx,ky1,nb)
    save(ph*"/ben"*string(nkx)*string(cc)*".jld2","ben",ben1d,"ky",ky1)
    println(time()-t,"s")
    return ben1d
end
#ben1d=maineig()



#ben1d=maineig()