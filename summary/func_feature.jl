function wavef_real(ev::Array{ComplexF64,1},kk::Array{Float64,1},Gkx::Array{Int,1},
    Gky::Array{Int,1},lenGk::Int,xx::Float64,yy::Float64,b1::Array{Float64,1},
    b2::Array{Float64,1})
    waveup::ComplexF64=0.0im
    wavedw::ComplexF64=0.0im
    for ii in 1:lenGk
        vec1,vec2=Gkx[ii].*b1.+Gky[ii].*b2.+kk
        wf=exp(1.0im*(vec1*xx+vec2*yy))
        waveup+=ev[ii]*wf
        wavedw+=ev[ii+lenGk]*wf
    end
    return waveup,wavedw
end

function wavedensity(xx::Array{Float64,1},yy::Array{Float64,1},kk::Array{Float64,1},
    ev::Array{ComplexF64,1},Gkx::Array{Int,1},Gky::Array{Int,1},
    lenGk::Int,b1::Array{Float64,1},b2::Array{Float64,1})

    lenxx=length(xx)
    lenyy=length(yy)
    wden=Array{Float64,3}(undef,4,lenxx,lenyy)
    for iy in 1:lenyy,ix in 1:lenxx
        tmp1,tmp2=wavef_real(ev,kk,Gkx,Gky,lenGk,xx[ix],yy[iy],b1,b2)
        wden[1,ix,iy]=angle(tmp1) #abs2(tmp1)+abs2(tmp2)
        wden[2,ix,iy]=angle(tmp2)
        wden[3,ix,iy]=abs2(tmp1)
        wden[4,ix,iy]=abs2(tmp2)
    end
    return wden
end
function wavespin(xx::Array{Float64,1},yy::Array{Float64,1},kk::Array{Float64,1},
    ev::Array{ComplexF64,1},Gkx::Array{Int,1},Gky::Array{Int,1},
    lenGk::Int,b1::Array{Float64,1},b2::Array{Float64,1})

    lenxx=length(xx)
    lenyy=length(yy)
    wden=Array{Float64,3}(undef,4,lenxx,lenyy)
    for iy in 1:lenyy,ix in 1:lenxx
        tmp1,tmp2=wavef_real(ev,kk,Gkx,Gky,lenGk,xx[ix],yy[iy],b1,b2)
        wden[1,ix,iy]=real(conj(tmp2)*tmp1+conj(tmp1)*tmp2)
        wden[2,ix,iy]=real((conj(tmp2)*tmp1-conj(tmp1)*tmp2)*1.0im)
        wden[3,ix,iy]=abs2(tmp1)-abs2(tmp2)
        wden[4,ix,iy]=abs2(tmp2)+abs2(tmp1)
    end
    return wden
end

function sg(ev::Array{ComplexF64,1})
    lenev=round(Int,length(ev)/2)
    sgz=sgx=sgy=0.0im
    for ii in 1:lenev
        sgx+=ev[ii]*conj(ev[ii+lenev])+conj(ev[ii])*ev[ii+lenev]
        sgy+=ev[ii]*conj(ev[ii+lenev])*1im-1im*conj(ev[ii])*ev[ii+lenev]
        sgz+=conj(ev[ii])*ev[ii]-conj(ev[ii+lenev])*ev[ii+lenev]
    end
    return real.([sgx,sgy,sgz])
end

function meanz(ev1::T,ev2::T) where T<:Array{ComplexF64,1}
    lenev=round(Int,length(ev1)/2)
    sgz::ComplexF64=0.0im
    for ii in 1:lenev
        sgz+=conj(ev1[ii])*ev2[ii]-conj(ev1[ii+lenev])*ev2[ii+lenev]
    end
    return sgz
end

function polarev(bev::Array{ComplexF64,4})
    lenGk,nb,lenkx,lenky=size(bev)
    #bev=SharedArray{ComplexF64,4}(lenGk,nb,lenkx,lenky)
    spg=SharedArray{Float64,4}(3,nb,lenkx,lenky)
    nb=round(Int,nb/2)
    for iy in 1:lenky
        @sync @distributed for ix in 1:lenkx
        for nn in 1:nb
            tmp1=bev[:,2*nn-1,ix,iy]
            tmp2=bev[:,2*nn,ix,iy]
            tmp1[:],tmp2[:]=sigmazilize(tmp1,tmp2)
            if real(meanz(tmp1,tmp1))>0
                #bev[:,2*nn-1,ix,iy].=tmp2
                #bev[:,2*nn,ix,iy].=tmp1
                spg[:,2*nn-1,ix,iy].=sg(tmp2)
                spg[:,2*nn,ix,iy].=sg(tmp1)
            else
                #bev[:,2*nn-1,ix,iy].=tmp1
                #bev[:,2*nn,ix,iy].=tmp2
                spg[:,2*nn-1,ix,iy].=sg(tmp1)
                spg[:,2*nn,ix,iy].=sg(tmp2)
            end
        end
        end
    end
    return spg
end