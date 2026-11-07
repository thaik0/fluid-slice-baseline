import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from model import FlowNet
for width in [16,32,64,128,192]:
    print(f'width={width}: {sum(p.numel() for p in FlowNet(3,width).parameters()):,} parameters (3 channels)')
