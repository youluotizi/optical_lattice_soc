using FFTW
using GR

# Set Time Parameters
t0=0
tf=5
dt=.01

# Set Space Grid Parameters
dx=.05
xmax=6
# xmin will be -xmax.  Making the situation symmetric

t=collect(t0:dt:tf)
x=collect(-xmax:dx:xmax)

nt=length(t)
N=length(x)

dk=2*π/(N*dx)
if iseven(N)
    k[collect(0:N/2) ; collect(-N/2:-1)]*dk
    else # N is odd
    k=[collect(0:(N-1)/2) ; collect(-(N-1)/2:-1)]*dk
end

Vx=.5*x.^2
Vk=.5*k.^2

Uxh=exp.(-Vx*dt/2)
Ux=exp.(-Vx*dt)
Uf=exp.(-Vk*dt)

ϕ(x)=π^(-.25)*exp(-x.^2/2)
ϕ1(x)=sqrt(2)*π^(-.25)*exp(-x.^2/2)*x
ϕ2(x)=1/sqrt(2)*π^(-.25)*exp(-x.^2/2)*(2*x^2-1)
ϕ4(x)=1/(2*sqrt(6)*π^(.25))*exp(-x.^2/2)*(4*x^4-12*x^2+3)

Ψ0=(1.0/(2*xmax))*ones(Complex{Float64},N)
Ψtrue=(1+0im).*ϕ.(x)
fΨtrue=fft(Ψtrue)
Ψ1=(1+0im).*ϕ1.(x)
Ψ2=(1+0im).*ϕ2.(x)
Ψ4=(1+0im).*ϕ4.(x)

plot(x,abs.(Ψ0),x,abs.(Ψtrue),x,abs.(Ψ1),x,abs.(Ψ2))

ft=plan_fft(Ψ0);
Ψf=ft*Ψ0;
ift=plan_ifft(Ψf);

## If you are doing lots of fft's, plan them before for optimization!
ft=plan_fft(Ψ0);
Ψf=ft*Ψ0;
ift=plan_ifft(Ψf);

nmeas=1
lmeas=floor(Int, nt/nmeas)

Ex=zeros(Float64,lmeas);
Ek=zeros(Float64,lmeas);
E=zeros(Float64,lmeas);
E2=zeros(Float64,lmeas);

c=zeros(Float64,lmeas);
c1=zeros(Float64,lmeas);
c2=zeros(Float64,lmeas);
c4=zeros(Float64,lmeas);


Ψ0=(1.0/(2*xmax))*ones(Complex{Float64},N);
Ψ=copy(Ψ0);
jj=1

# The operators we have to start off with
Ψ=Ψ.*Uxh

Psif=(ft*Ψ)*dx
Psif=Psif.*Uf    
Ψ=(ift*Psif)*(dk/2π)


function main(nt,Ψ,Ux)
    for ii in 1:nt

        Ψ=Ψ.*Ux
        Psif=(ft*Ψ)*dx
        Psif=Psif.*Uf

        Ψ=(ift*Psif)*(dk/2π)
        nn=sum(conj(Ψ).*Ψ)*dx
        Ψ=1/sqrt(nn)*Ψ

        if ii%nmeas == 0
            # Every time we measure, we have to finish with a U_x half time step
            Ψt=Ψ.*Uxh
            Ψft=(ft*Ψt)*dx

            Ex[jj]=real(sum(conj(Ψt).*Vx.*Ψt))*dx

            Ek[jj]=real(sum(conj(Ψft).*Vk.*Ψft))*(dk/2π)

            E[jj]=Ex[jj]+Ek[jj]
            c[jj]=abs(sum( conj(Ψt).* Ψtrue)) *dx
            c1[jj]=abs(sum( conj(Ψt).* Ψ1)) *dx
            c2[jj]=abs(sum( conj(Ψt).* Ψ2)) *dx
            c4[jj]=abs(sum( conj(Ψt).* Ψ4)) *dx

            jj+=1
        end
    end
end


Ψ=Ψ.*Uxh;
fΨ=ft*Ψ;

plot(x,abs.(Ψ),x,abs.(Ψtrue))

function test()
    a=Array{Float64,2}(undef,10,10)
    for ii in 1:10,jj in 1:10
        a[jj,ii]=sin(ii*2*pi/10)+cos(ii*2*pi/10)
    end
    return a
end
