using LinearAlgebra
using GR

function Caloffm(Gkx::Array{Int,1},Gky::Array{Int,1},m0::Float64,
    v0::Float64,lenGk::Int,t::Float64)
    mat=zeros(ComplexF64,2*lenGk,2*lenGk);
    #mat_ky=Diagonal(ones(nky))
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

function enband1D(mat::Array{ComplexF64,2},Gkx::Array{Int,1},Gky::Array{Int,1},
    lenGk::Int,kx::Array{Float64,1},ky::Array{Float64,1},b1::Array{Float64,1},
    b2::Array{Float64,1},v0::Float64,nb::Int,t::Float64)

    lenkx=length(kx)
    en=Array{Float64,2}(undef,nb,lenkx)
    ev=Array{ComplexF64,3}(undef,2*lenGk,nb,lenkx)

    for ik in 1:lenkx
        for mm in 1:lenGk
            vec_tmp=[kx[ik],ky[ik]]+Gkx[mm]*b1+Gky[mm]*b2
            mat[2*mm-1,2*mm-1]=mat[2*mm,2*mm]=dot(vec_tmp,vec_tmp)+v0
        end
        #en[:,ikx,iky],ev[:,:,ikx,iky]=eigen(Hermitian(mat),1:nb)
        #en_tmp,ev_tmp=eigen(Hermitian(mat))
        en_tmp=eigvals(Hermitian(mat))
        pt=partialsortperm(en_tmp,1:nb)
        en[:,ik]=en_tmp[pt]
        #ev[:,:,ikx]=ev_tmp[:,pt]
    end
    println(time()-t,"s single particle 1D band")
    return en#,ev
end

function main1d()
    t=time()
    Gmax=6
    gg,fg=0.2,1.0
    optn=90
    v0,m0,nb=4.0,0.0,4
    nk1d=80
    lenb1=sqrt(2)
    lenb2=sqrt(2)
    b1,b2=[lenb1,0.0],[0.0,lenb2]

    ph=pathout(Gmax,gg,fg,optn,1)
    #ph=1

    #kx,ky=symline([-lenb1/2 0;lenb1/2 0],nk1d,t)
    kx,ky,rr=linsym([0 0;lenb1/2 0;lenb1/2 lenb2/2;0 0],nk1d,t)
    #kx,ky=bz2d([-lenb1/2 0;lenb1/2 0],nk1d,0.1,t)
    #kx=[0.0];ky=[0.0]

    Gkx,Gky,lenGk=CalGk(b1,b2,Gmax)
    mat=Caloffm(Gkx,Gky,m0,v0,lenGk,t)
    en1d=enband1D(mat,Gkx,Gky,lenGk,kx,ky,b1,b2,v0,nb,t)
    #ind,lenind=myind(Gkx,Gky,lenGk,t)
    #ben1d=eigBdgM1D(ind,lenind,mat,lenGk,Gkx,Gky,b1,b2,gg,v0,nb,fg,kx,ky,ph,t)
    return en1d,rr
end
