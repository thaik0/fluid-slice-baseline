"""Small multiscale velocity regressor; original implementation, no pretrained weights."""
import torch
from torch import nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, a, b):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(a,b,3,padding=1), nn.GroupNorm(4,b), nn.SiLU(), nn.Conv2d(b,b,3,padding=1), nn.GroupNorm(4,b), nn.SiLU())
    def forward(self,x): return self.net(x)


class FlowNet(nn.Module):
    def __init__(self, channels=3, width=32):
        super().__init__()
        if width % 4: raise ValueError('width must be divisible by four')
        self.a = Block(channels+1,width)
        self.b = Block(width,2*width)
        self.c = Block(2*width,4*width)
        self.d = Block(6*width,2*width)
        self.e = Block(3*width,width)
        self.out = nn.Conv2d(width,channels,1)
    def forward(self,x,t):
        t = torch.as_tensor(t,device=x.device,dtype=x.dtype).reshape(-1,1,1,1)
        t = t.expand(x.shape[0],1,*x.shape[-2:])
        a = self.a(torch.cat([x,t],1)); b = self.b(F.avg_pool2d(a,2)); c = self.c(F.avg_pool2d(b,2))
        d = self.d(torch.cat([F.interpolate(c,size=b.shape[-2:],mode='nearest'),b],1))
        return self.out(self.e(torch.cat([F.interpolate(d,size=a.shape[-2:],mode='nearest'),a],1)))


def tiled(model,x,t,patch,shift=(0,0),batch=8):
    n,c,h,w = x.shape
    if h%patch or w%patch or patch%4 or batch<1: raise ValueError('Invalid patch/grid/batch')
    y = torch.roll(x,shift,(-2,-1))
    p = y.reshape(n,c,h//patch,patch,w//patch,patch).permute(0,2,4,1,3,5).reshape(-1,c,patch,patch)
    q = torch.cat([model(v,t) for v in p.split(batch)])
    y = q.reshape(n,h//patch,w//patch,c,patch,patch).permute(0,3,1,4,2,5).reshape(n,c,h,w)
    return torch.roll(y,tuple(-s for s in shift),(-2,-1))


@torch.no_grad()
def sample(model,noise,steps,patch,shifted,seed,batch=8):
    if steps<1: raise ValueError('steps must be positive')
    rng = torch.Generator().manual_seed(seed)
    x = noise.clone(); shifts = []
    for i in range(steps):
        shift = tuple(torch.randint(patch,(2,),generator=rng).tolist()) if shifted else (0,0)
        shifts.append(shift)
        x = x + tiled(model,x,i/steps,patch,shift,batch)/steps
    return x, shifts
