"""Small multiscale velocity regressor; original implementation, no pretrained weights."""
import torch
from torch import nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, a, b, normalization="group", padding="zeros"):
        super().__init__()
        norm = lambda: nn.GroupNorm(4,b) if normalization == "group" else nn.Identity()
        self.net = nn.Sequential(nn.Conv2d(a,b,3,padding=1,padding_mode=padding), norm(), nn.SiLU(), nn.Conv2d(b,b,3,padding=1,padding_mode=padding), norm(), nn.SiLU())
    def forward(self,x): return self.net(x)


class FlowNet(nn.Module):
    def __init__(self, channels=3, width=32, normalization="group", padding="zeros", mean_mode="implicit"):
        super().__init__()
        if mean_mode not in ["implicit","learned"]: raise ValueError("Unknown mean mode")
        self.mean_mode = mean_mode
        if normalization not in ['group','none'] or padding not in ['zeros','circular']:
            raise ValueError('Unknown normalization or padding')
        if width % 4: raise ValueError('width must be divisible by four')
        self.a = Block(channels+1,width,normalization,padding)
        self.b = Block(width,2*width,normalization,padding)
        self.c = Block(2*width,4*width,normalization,padding)
        self.d = Block(6*width,2*width,normalization,padding)
        self.e = Block(3*width,width,normalization,padding)
        self.out = nn.Conv2d(width,channels,1)
        if mean_mode == 'learned':
            # Preserve the parent's RNG stream: backbone weights and subsequent
            # training noise/time draws match the implicit-mean control.
            with torch.random.fork_rng(devices=[]):
                self.mean_head = nn.Sequential(nn.Linear(channels+1,64),nn.SiLU(),nn.Linear(64,64),nn.SiLU(),nn.Linear(64,channels))
    def forward(self,x,t):
        t = torch.as_tensor(t,device=x.device,dtype=x.dtype).reshape(-1,1,1,1)
        t = t.expand(x.shape[0],1,*x.shape[-2:])
        a = self.a(torch.cat([x,t],1)); b = self.b(F.avg_pool2d(a,2)); c = self.c(F.avg_pool2d(b,2))
        d = self.d(torch.cat([F.interpolate(c,size=b.shape[-2:],mode='nearest'),b],1))
        spatial = self.out(self.e(torch.cat([F.interpolate(d,size=a.shape[-2:],mode='nearest'),a],1)))
        if self.mean_mode == 'implicit': return spatial
        avg = self.mean_head(torch.cat([x.mean((-2,-1)),t[:,:,0,0]],1))[:,:,None,None]
        return spatial-spatial.mean((-2,-1),keepdim=True)+avg


def tiled(model,x,t,patch,shift=(0,0),batch=8):
    n,c,h,w = x.shape
    validate_domain('zeros',patch,(h,w),getattr(model,'mean_mode','implicit'))
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


def validate_domain(padding, patch, shape, mean_mode="implicit"):
    """A periodic whole plane may wrap; an arbitrary local crop may not."""
    if mean_mode == 'learned' and tuple(shape) != (patch,patch):
        raise ValueError('Learned mean branch requires full square planes, not local crop means')
    if padding == 'circular' and tuple(shape) != (patch,patch):
        raise ValueError('Circular padding requires full square periodic planes, not local crops')
