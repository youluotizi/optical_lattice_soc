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

####################################################
#         the point in reciprocal lattice vector
####################################################
function CalGk(b1::Array{Float64,1},b2::Array{Float64,1},Gmax::Int)
    Gkx=Array{Int,1}(undef,(2*Gmax)^2)
    Gky=Array{Int,1}(undef,(2*Gmax)^2)
    kk::Int=0
    lenb=max(norm(b1),norm(b2))

    pp=Gmax*lenb+1e-5
    for jj in -Gmax:Gmax,ii in -Gmax:Gmax
        if norm(ii .* b1 .+ jj .* b2)<pp
            kk+=1
            Gkx[kk]=ii
            Gky[kk]=jj
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
        lenpath+=norm(p[:,ii+1].-p[:,ii])
    end
    delta=lenpath/(num-1)
    rr=Float64[0.0]
    r0::Float64=0.0
    bz=Array{Float64,2}(undef,2,num+20)
    kk::Int=1
    bz[:,1].=p[:,1]
    for ii in 1:lenp-1
        p0=p[:,ii]
        vc=p[:,ii+1].-p0
        lenvc=norm(vc)
        vc.=vc./lenvc
        for jj in 1:num
            if jj*delta>lenvc-0.2*delta
                kk+=1
                bz[:,kk].=p[:,ii+1]
                r0+=lenvc-(jj-1)*delta
                push!(rr,r0)
                break
            end
            kk+=1
            bz[:,kk].=p0 .+(jj*delta).*vc
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
    xtmp,_=symline(pointlist,num)
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
    en=SharedArray{Float64,3}(nb,lenkx,lenky)
    ev=SharedArray{ComplexF64,4}(2*lenGk,nb,lenkx,lenky)

    for iky in 1:lenky
        @sync @distributed for ikx in 1:lenkx
            ktmp=[kx[ikx],ky[iky]]
            @inbounds for mm in 1:lenGk
                vec_tmp = ktmp.+Gkx[mm].*b1.+Gky[mm].*b2
                tmp=dot(vec_tmp,vec_tmp)+v0
                mat[mm,mm]=tmp+mz
                mat[mm+lenGk,mm+lenGk]=tmp-mz
            end
            #en_tmp=eigvals(Hermitian(mat))
            en_tmp,ev_tmp=eigen(Hermitian(mat))
            pt=partialsortperm(en_tmp,1:nb)
            en[:,ikx,iky].=en_tmp[pt]
            ev[:,:,ikx,iky].=ev_tmp[:,pt]
        end
    end
    return Array(en),Array(ev)
end

function enband1D(mat::Array{ComplexF64,2},Gkx::Array{Int,1},Gky::Array{Int,1},
    lenGk::Int,kx::Array{Float64,1},ky::Array{Float64,1},b1::Array{Float64,1},
    b2::Array{Float64,1},v0::Float64,mz::Float64,nb::Int)

    lenkx = length(kx)
    en = Array{Float64,2}(undef,nb,lenkx)
    #ev = Array{ComplexF64,3}(undef,2*lenGk,nb,lenkx)
    vec_tmp=Array{Float64,1}(undef,2)
    ktmp=Array{Float64,1}(undef,2)
    @inbounds for ik in 1:lenkx
        ktmp.=[kx[ik],ky[ik]]
        for mm in 1:lenGk
            vec_tmp .= ktmp.+Gkx[mm].*b1 .+Gky[mm].*b2
            tmp = dot(vec_tmp,vec_tmp)+v0
            mat[mm,mm] = tmp+mz
            mat[mm+lenGk,mm+lenGk] = tmp-mz
        end
        en_tmp = eigvals(Hermitian(mat))
        pt = partialsortperm(en_tmp,1:nb)
        en[:,ik] .= en_tmp[pt]
    end
    return en#,ev
end

####################################################
#      the index result to phi-4 not integrated to zero
####################################################
function myidx(Gkx::Array{Int,1},Gky::Array{Int,1},lenGk::Int)
    idx=Array{Int,2}(undef,4,lenGk^3*floor(Int,lenGk/2))
    kk::Int=0
    @inbounds for ii in 1:lenGk,jj in 1:lenGk,mm in 1:lenGk
        tx = -Gkx[ii]-Gkx[jj]+Gkx[mm]
        ty = -Gky[ii]-Gky[jj]+Gky[mm]
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

#####################################################
#           the interaction part of Bdg matrix
#####################################################
function mintwo(ph::String,lenGk::Int,idx::Int)
    evtmp=load(ph*"/ev0.jld2","ev0")
    u0=load(ph*"/ev0.jld2","u0")::Float64
    ev0=Array{ComplexF64,1}(undef,2*lenGk)
    ev0.=evtmp[:,idx]
    return ev0,u0
end

function intBdgM!(matH::Array{ComplexF64,2},ev0::Array{ComplexF64,1},ind::Array{Int,2},
    lenind::Int,lenGk::Int,u0::Float64,guu::Float64,gud::Float64,gdd::Float64)

    lenmat::Int = 2*lenGk
    @inbounds for ii in 1:lenind
        t1,t2,t3,t4 = view(ind,:,ii)
        #
        tmp::ComplexF64 = 2*conj(ev0[t2])*ev0[t3]*guu
        tmp += conj(ev0[t2+lenGk])*ev0[t3+lenGk]*gud
        matH[t1,t4] += tmp
        matH[t1+lenmat,t4+lenmat] += conj(tmp)
        #
        tmp = conj(ev0[t2+lenGk])*ev0[t3]*gud
        matH[t1,t4+lenGk] += tmp
        matH[t1+lenmat,t4+lenGk+lenmat] += conj(tmp)

        tmp = conj(ev0[t2])*ev0[t3+lenGk]*gud
        matH[t1+lenGk,t4] += tmp
        matH[t1+lenGk+lenmat,t4+lenmat] += conj(tmp)

        tmp = 2*conj(ev0[t2+lenGk])*ev0[t3+lenGk]*gdd+conj(ev0[t2])*ev0[t3]*gud
        matH[t1+lenGk,t4+lenGk] += tmp
        matH[t1+lenGk+lenmat,t4+lenGk+lenmat] += conj(tmp)
        #
        ############# off diag #####################
        tmp = ev0[t3]*ev0[t4]*guu
        matH[t1,t2+lenmat] += tmp
        matH[t2+lenmat,t1] += conj(tmp)

        tmp = ev0[t3+lenGk]*ev0[t4]*gud
        matH[t1,t2+lenGk+lenmat] += tmp
        matH[t2+lenGk+lenmat,t1] += conj(tmp)

        tmp = ev0[t3]*ev0[t4+lenGk]*gud
        matH[t1+lenGk,t2+lenmat] += tmp
        matH[t2+lenmat,t1+lenGk] += conj(tmp)

        tmp = ev0[t3+lenGk]*ev0[t4+lenGk]*gdd
        matH[t1+lenGk,t2+lenGk+lenmat] += tmp
        matH[t2+lenGk+lenmat,t1+lenGk] += conj(tmp)
        #
    end
    #
    for ii in 1:2*lenmat
        matH[ii,ii]-=u0
    end
    #
    nothing
end

#####################################################
#               the Bdg matrix
#####################################################
function matHU0!(matH::Array{ComplexF64,2},mat::Array{ComplexF64,2},lenGk::Int,
    v0::Float64,b1::Array{Float64,1},b2::Array{Float64,1},Gkx::Array{Int,1},
    Gky::Array{Int,1},mz::Float64,pband::Int)
    lenmat=2*lenGk
    tauz=Diagonal([ones(ComplexF64,lenmat); fill(-1.0+0.0im,lenmat)])
    @inbounds for ii in 1:2*lenGk
        mat[ii,ii]=0.0
    end

    matH[1:lenmat,1:lenmat].+=mat./2
    matH[lenmat+1:end,lenmat+1:end].+=conj.(mat)./2

    mtmp=copy(matH)
    vtmp=Array{Float64,1}(undef,2)
    @inbounds for mm in 1:lenGk
        vtmp.=Gkx[mm].*b1.+Gky[mm].*b2
        tmp=dot(vtmp,vtmp)+v0
        mtmp[mm,mm]+=(tmp+mz)/2
        mtmp[mm+lenGk,mm+lenGk]+=(tmp-mz)/2

        mtmp[mm+lenmat,mm+lenmat]+=(tmp+mz)/2
        mtmp[mm+lenGk+lenmat,mm+lenGk+lenmat]+=(tmp-mz)/2
    end

    ben=eigvals(Hermitian(mtmp),1:20)
    u0=ben[1+2*pband]
    println("delta_u0:",u0)
    #
    @inbounds for mm in 1:2*lenmat
        matH[mm,mm]-=u0
    end
    #
    lmul!(tauz,matH)
end

@everywhere function BdgM2d(matH::Array{ComplexF64,2},lenGk::Int,
    b1::Array{Float64,1},b2::Array{Float64,1},v0::Float64,Gkx::Array{Int,1},
    Gky::Array{Int,1},kx::Float64,ky::Float64,nb::Int,mz::Float64)

    lenmat=2*lenGk
    vtmp=Array{Float64,1}(undef,2)
    kxy=[kx,ky]
    @inbounds for mm in 1:lenGk
        vtmp .= kxy.+Gkx[mm].*b1 .+Gky[mm].*b2
        tmp=dot(vtmp,vtmp)+v0
        matH[mm,mm]+=(tmp+mz)/2
        matH[mm+lenGk,mm+lenGk]+=(tmp-mz)/2

        vtmp .= (-1).*kxy.+ Gkx[mm].*b1 .+Gky[mm].*b2
        tmp=dot(vtmp,vtmp)+v0
        matH[mm+lenmat,mm+lenmat]-=(tmp+mz)/2
        matH[mm+lenGk+lenmat,mm+lenGk+lenmat]-=(tmp-mz)/2
    end
    ben,bev=eigen(matH)
    pt=partialsortperm(real(ben),1:lenmat+nb)
    return ben[pt[lenmat+1:end]],bev[:,pt[lenmat+1:end]]
end

@everywhere function BdgM1d(matH::Array{ComplexF64,2},lenGk::Int,
    b1::Array{Float64,1},b2::Array{Float64,1},v0::Float64,Gkx::Array{Int,1},
    Gky::Array{Int,1},kx::Float64,ky::Float64,nb::Int,mz::Float64)

    lenmat = 2*lenGk
    vtmp=Array{Float64,1}(undef,2)
    kxy=[kx,ky]
    @inbounds for mm in 1:lenGk
        vtmp .= kxy.+Gkx[mm].*b1 .+Gky[mm].*b2
        tmp = dot(vtmp,vtmp)+v0
        matH[mm,mm] += (tmp+mz)/2
        matH[mm+lenGk,mm+lenGk] += (tmp-mz)/2

        vtmp .= (-1).*kxy.+Gkx[mm].*b1 .+Gky[mm].*b2
        tmp = dot(vtmp,vtmp)+v0
        matH[mm+lenmat,mm+lenmat] -= (tmp+mz)/2
        matH[mm+lenGk+lenmat,mm+lenGk+lenmat] -= (tmp-mz)/2
    end
    ben = eigvals(matH)
    pt = partialsortperm(real(ben),1:lenmat+nb)
    return ben[pt[lenmat+1:end]]
end

#####################################################
#              eigen Bdg
#####################################################
function eigBdgM1D(matH::Array{ComplexF64,2},lenGk::Int,Gkx::Array{Int,1},
    Gky::Array{Int,1},b1::Array{Float64,1},b2::Array{Float64,1},v0::Float64,nb::Int,
    kx::Array{Float64,1},ky::Array{Float64,1},mz::Float64)

    lenkx=length(kx)
    ben_tmp=pmap((ix,iy)->BdgM1d(matH[:,:],lenGk,b1,b2,v0,Gkx,Gky,ix,iy,nb,mz),kx,ky)
    ben=Array{ComplexF64,2}(undef,nb,lenkx)
    for ikx in 1:lenkx
        ben[:,ikx].=ben_tmp[ikx]
    end
    println("Check_ben:",sum(abs.(imag.(ben))))
    return real.(ben)
end

function eigBdgM2D(matH::Array{ComplexF64,2},lenGk::Int,Gkx::Array{Int,1},
    Gky::Array{Int,1},b1::Array{Float64,1},b2::Array{Float64,1},v0::Float64,
    nb::Int,kx::Array{Float64,1},ky::Array{Float64,1},mz::Float64)

    lenkx=length(kx)
    lenky=length(ky)

    kxlist=Array{Float64,1}(undef,lenkx*lenky)
    kylist=Array{Float64,1}(undef,lenkx*lenky)
    kk::Int=0
    for iky in 1:lenky,ikx in 1:lenkx
        kk+=1
        kylist[kk]=ky[iky]
        kxlist[kk]=kx[ikx]
    end
    ben_tmp=pmap((ix,iy)->BdgM2d(matH[:,:],lenGk,b1,b2,v0,Gkx,Gky,ix,iy,nb,mz),kxlist,kylist)

    ben=Array{ComplexF64,3}(undef,nb,lenkx,lenky)
    bev=Array{ComplexF64,4}(undef,4*lenGk,nb,lenkx,lenky)
    kk=0
    for iky in 1:lenky,ikx in 1:lenkx
        kk+=1
        ben[:,ikx,iky],bev[:,:,ikx,iky]=ben_tmp[kk]
    end
    println("check_band:",sum(abs.(imag.(ben))))
    return real.(ben),bev
end

#####################################################
#             Berry curvature and Chern number
#####################################################
function Bcuvat(bev::Array{ComplexF64,4})
    lenev,mid,lenkx,lenky=size(bev)
    lenev=round(Int,lenev/2)
    Chern=Array{Float64,1}(undef,mid)
    bcav=Array{Float64,3}(undef,mid,lenkx-1,lenky-1)
    tauz=Diagonal([ones(ComplexF64,lenev); -ones(ComplexF64,lenev)])
    ev=bev
    for iy in 1:lenky,ix in 1:lenkx,ii in 1:mid
        ev_tmp=view(ev,:,ii,ix,iy)
        tmp=sqrt(ev_tmp'*tauz*ev_tmp)
        ev[:,ii,ix,iy].=ev[:,ii,ix,iy]./tmp
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
        @views for ii in 1:mid
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
        Chern[ii]=tmp/(2*pi)
    end
    return Chern,bcav
end

#####################################################
#             part ben and export
#####################################################
function myexport(t::Float64,ben2d::Array{Float64,3},bcav::Array{Float64,3},
    chern::Array{Float64,1},guu::Float64,gud::Float64,gdd::Float64,ph::String,mindx::Int)
    #=
    try
        mkdir(ph)
    catch
        nothing
    end
    
    for ii in 1:nb
        writedlm(ph*"/ben"*string(ii),real(ben2d[ii,:,:]))
    end

    for ii in 1:nb
        writedlm(ph*"/bcav"*string(ii),bcav[ii,:,:])
    end
    =#
    save(ph*"/bcav"*string(mindx)*".jld2","ben2d",real.(ben2d),"bcav",bcav)
    para=[guu,gud,gdd]
    paraname="guu,gud,gdd"
    writedlm(ph*"/Chern"*string(mindx),
    [[string(now()),"time used:",time()-t,paraname,para,"Chern="];chern])
end


function myplot(rr::Array{Float64,1},ben::Array{Float64,2})
    lenben=size(ben,1)
    data=[rr,ben[1,:]]
    for ii in 2:lenben
        push!(data,rr)
        push!(data,ben[ii,:])
    end
    GR.plot(data...)
    nothing
end

#=
function myload(theta::Float64,ind::Array{Int64,2},lenind::Int,
    guu::Float64,gdd::Float64,gud::Float64,path::String)
    tmp=load(path)
    ev=tmp["ev"]
    en=tmp["en"]
    xopt=tmp["xopt"]
    lenGk,optn=size(ev)
    ev0=zeros(ComplexF64,lenGk)
    xopt[2]*=exp(1.0im*theta)
    for ii in 1:optn
        ev0.+=ev[:,ii].*xopt[ii]
    end
    lenGk=round(Int,lenGk/2)
    u0=0.0im
    @inbounds for ii in 1:lenind
        t1,t2,t3,t4 = view(ind,:,ii)
        u0 += conj(ev0[t1]*ev0[t2])*ev0[t3]*ev0[t4]*guu
        u0 += conj(ev0[t1+lenGk]*ev0[t2+lenGk])*ev0[t3+lenGk]*ev0[t4+lenGk]*gdd
        u0 += conj(ev0[t1]*ev0[t2+lenGk])*ev0[t3+lenGk]*ev0[t4]*gud
        u0 += conj(ev0[t1+lenGk]*ev0[t2])*ev0[t3]*ev0[t4+lenGk]*gud
    end
    return real(u0)+en/2,ev0
end


###########################################
#           test part
############################################

function testmain()
    println("--------------Pexp------------")
    t=time()
    Gmax,nb,optn=5,4,2
    m0,mz=2.0,0.0
    nk1d=180
    v0,guu,gdd,gud=2.0,0.174868875,0.174868875,0.17446275
    #guu=0.175275 gdd=0.17446275
    
    lenb1,lenb2=sqrt(2),sqrt(2)
    b1,b2=[lenb1,0.0],[0.0,lenb2]
    Gkx,Gky,lenGk=CalGk(b1,b2,Gmax)
    mat=Caloffm(Gkx,Gky,m0,v0,lenGk)
    kx,ky,rr=linsym([0 0;lenb1/2 0;lenb1/2 lenb2/2;0 0],nk1d)
 
    ph=pathout(Gmax,v0,m0,mz,optn)
    #return enband1D(mat,Gkx,Gky,lenGk,kx,ky,b1,b2,v0,mz,nb)
    ind,lenind=myind(Gkx,Gky,lenGk)
    println("lenind:",lenind)

    ev,ev0,xopt,en0,uu0=maincoetest(Gkx,Gky,m0,v0,lenGk,b1,b2,optn,ind,lenind,guu,gud,gdd,mz)
    return real.(eigBdgM1D(ind,lenind,mat,lenGk,Gkx,Gky,b1,b2,guu,gud,gdd,v0,nb,kx,ky,real(uu0),ev0,mz))
    
    save(ph*"/ev.jld2", "en",en0,"u0",uu0,"ev",ev,"xopt",xopt,"mat",mat)
    return nothing
    θ=collect(0.0:2000*pi/40:2*pi-1e-6)
    lenth=length(θ)
    uout=Array{Float64,1}(undef,lenth)
    sgout=Array{Float64,2}(undef,3,lenth)
    ben1d=Array{Float64,3}(undef,nb,length(kx),lenth)
    for jj in 1:lenth
        ev0.=0.0im
        dtmp=copy(xopt)
        dtmp[2]*=exp(1.0im*θ[jj])

        for ii in 1:optn
            ev0.+=ev[:,ii].*dtmp[ii]
        end
        u0::ComplexF64=0.0im
        @inbounds for ii in 1:lenind
            t1,t2,t3,t4 = view(ind,:,ii)
            u0 += conj(ev0[t1]*ev0[t2])*ev0[t3]*ev0[t4]*guu
            u0 += conj(ev0[t1+lenGk]*ev0[t2+lenGk])*ev0[t3+lenGk]*ev0[t4+lenGk]*gdd
            u0 += conj(ev0[t1]*ev0[t2+lenGk])*ev0[t3+lenGk]*ev0[t4]*gud
            u0 += conj(ev0[t1+lenGk]*ev0[t2])*ev0[t3]*ev0[t4+lenGk]*gud
        end
        ben1d[:,:,jj].=real.(eigBdgM1D(ind,lenind,mat,lenGk,Gkx,Gky,b1,b2,guu,gud,gdd,v0,nb,kx,ky,real(u0+en0/2),ev0,mz))
        uout[jj]=real(u0)+en0
        sgout[:,jj].=sg(ev0)
    end
    println(time()-t,"s end")
    save(ph*"/mz.jld2","ben1d",ben1d,"sg",sgout,"u0",uout)
    return ben1d,sgout,uout
end
#ben1d=testmain()

function mainpolar()
    t=time()
    Gmax,nb,optn=6,2,2
    v0,guu,gdd,gud=2.0,0.174868875,0.174868875,0.17446275 #guu=0.175275 gdd=0.17446275
    m0,mz=1.0,0.0
    nk2d=15
    lenb1,lenb2=sqrt(2),sqrt(2)
    b1,b2=[lenb1,0.0],[0.0,lenb2]

    ph=pathout(Gmax,v0,m0,mz,optn)
    Gkx,Gky,lenGk=CalGk(b1,b2,Gmax)
    mat=Caloffm(Gkx,Gky,m0,v0,lenGk)
    kx,ky=bz2d([-lenb1/2 -lenb2/2;lenb1/2 lenb2/2],nk2d,0.0)
    ben2d,bev2d=enband2D(mat,Gkx,Gky,lenGk,kx,ky,b1,b2,v0,mz,nb)
    println(time()-t," eig end")
    spg=polarev(bev2d)
    println(time()-t," end")
    return ben2d,spg
end

function findgap(ben::Array{Float64,3})
    nb,lenkx,lenth=size(ben)
    gap=Array{Float64,1}(undef,lenth)
    gappt=Array{Int,1}(undef,lenth)
    for ith in 1:lenth
        tmp=1.0
        pt=1
        band1=ben[2,:,ith]
        band2=ben[3,:,ith]
        for ik in 1:lenkx
            if tmp>band2[ik]-band1[ik]
                tmp=band2[ik]-band1[ik]
                pt=ik
            end
        end
        gap[ith]=tmp
        gappt[ith]=pt
    end
    return gap,gappt
end
function findgap(ben::Array{Float64,2})
    lenkx=size(ben,2)
    tmp::Float64=10.0
    pt::Int=1
    for ik in 1:lenkx
        if tmp>ben[3,ik]-ben[2,ik]
            tmp=ben[3,ik]-ben[2,ik]
            pt=ik
        end
    end
    return tmp,pt
end
=#