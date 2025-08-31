

## 2. `poem_overfit.py`


import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------- 数据 ----------------
# 用一首诗：《静夜思》
poem = "床前明月光，疑是地上霜。举头望明月，低头思故乡。"

vocab = sorted(list(set(poem)))
stoi = {ch:i for i,ch in enumerate(vocab)}
itos = {i:ch for ch,i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(poem), dtype=torch.long)

block_size, batch_size = 8, 1   # 很小的上下文和批次
def get_batch():
    ix = torch.randint(len(data)-block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

# ---------------- 模型 ----------------
class Attention(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.qkv = nn.Linear(d_model, 3*d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B,T,C = x.size()
        qkv = self.qkv(x).view(B,T,self.n_head,3*self.d_head)
        q,k,v = qkv.chunk(3, dim=-1)
        q,k,v = [t.transpose(1,2) for t in (q,k,v)]
        att = (q @ k.transpose(-2,-1)) / (self.d_head**0.5)
        mask = torch.tril(torch.ones(T,T)).to(x.device)
        att = att.masked_fill(mask==0, -1e9)
        out = (att.softmax(-1) @ v).transpose(1,2).reshape(B,T,C)
        return self.proj(out)

class Block(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        self.attn = Attention(d_model, n_head)
        self.ff   = nn.Sequential(nn.Linear(d_model,4*d_model), nn.ReLU(), nn.Linear(4*d_model,d_model))
        self.ln1, self.ln2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
    def forward(self,x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class TinyGPT(nn.Module):
    def __init__(self, vocab_size, d_model=64, n_layer=2, n_head=2, block_size=8):
        super().__init__()
        self.token = nn.Embedding(vocab_size, d_model)
        self.pos   = nn.Embedding(block_size, d_model)
        self.blocks= nn.Sequential(*[Block(d_model,n_head) for _ in range(n_layer)])
        self.ln    = nn.LayerNorm(d_model)
        self.head  = nn.Linear(d_model, vocab_size)
        self.block_size = block_size
    def forward(self, idx):
        B,T = idx.size()
        x = self.token(idx) + self.pos(torch.arange(T, device=idx.device))
        return self.head(self.ln(self.blocks(x)))

# ---------------- 训练 ----------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model = TinyGPT(len(vocab)).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=1e-2)

for step in range(2000):  # 迭代多一点，容易记住
    xb,yb = get_batch()
    xb,yb = xb.to(device), yb.to(device)
    logits = model(xb)
    loss = F.cross_entropy(logits.view(-1,logits.size(-1)), yb.view(-1))
    opt.zero_grad(); loss.backward(); opt.step()
    if step%200==0: print(f"step {step}: loss {loss.item():.3f}")

# ---------------- 生成 ----------------
def generate(start="床前明月光", max_new=20):
    idx = torch.tensor([stoi[c] for c in start],dtype=torch.long)[None,:].to(device)
    for _ in range(max_new):
        logits = model(idx[:,-model.block_size:])
        probs  = F.softmax(logits[:,-1,:], dim=-1)
        next_id = torch.multinomial(probs,1)
        idx = torch.cat([idx,next_id],dim=1)
    return decode(idx[0].tolist())

print("生成示例：", generate("床前明月光"))
