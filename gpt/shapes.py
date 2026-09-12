"""Watch the shapes move through attention. No learning, no training - just
tensors changing shape, printed at every step.

    python gpt/shapes.py

Deliberately tiny numbers so every shape is readable. If the reshaping in
model.py TODO 1 feels abstract, run this first. It is the same sequence of
operations on numbers small enough to hold in your head.
"""

import torch

B, T, C, H = 2, 4, 6, 2   # batch, time, channels, heads
hs = C // H               # head size

print(f"B={B} sequences, T={T} tokens each, C={C} channels, H={H} heads, hs={hs}\n")

x = torch.randn(B, T, C)
print(f"x                                    {tuple(x.shape)}   the residual stream")

# One fused projection produces q, k and v together.
c_attn = torch.nn.Linear(C, 3 * C, bias=False)
qkv = c_attn(x)
print(f"c_attn(x)                            {tuple(qkv.shape)}   q, k and v end to end")

q, k, v = qkv.split(C, dim=2)
print(f"  .split(C, dim=2)                   {tuple(q.shape)} x3  <- 'three of (B, T, C)'")
print()

# Split the channel dim into heads. view() is free - no data moves, PyTorch
# just reinterprets the same memory with different strides.
q4 = q.view(B, T, H, hs)
print(f"q.view(B, T, H, hs)                  {tuple(q4.shape)}   channels split into heads")
print(f"  same memory?                       {q4.data_ptr() == q.data_ptr()}   <- view is free")

qh = q4.transpose(1, 2)
kh = k.view(B, T, H, hs).transpose(1, 2)
vh = v.view(B, T, H, hs).transpose(1, 2)
print(f"  .transpose(1, 2)                   {tuple(qh.shape)}   head dim moved beside batch")
print(f"  still same memory?                 {qh.data_ptr() == q.data_ptr()}   <- transpose only restrides")
print(f"  contiguous in memory?              {qh.is_contiguous()}  <- why .contiguous() is needed later")
print()

# THE reason for the transpose: matmul treats every dim except the last two
# as batch, and runs the matrix op independently across all of them.
att = qh @ kh.transpose(-2, -1)
print(f"q @ k.transpose(-2, -1)              {tuple(att.shape)}   B*H = {B*H} independent T x T score grids")
print(f"  the matrix part is the last two:   ({T}, {T})      every token scored against every token")
print()

tril = torch.tril(torch.ones(T, T))
print("causal mask (1 = may attend):")
print(tril.to(torch.int))
att = att.masked_fill(tril == 0, float("-inf"))
print("\nhead 0 of sequence 0, after masking - note -inf above the diagonal:")
print(att[0, 0])

att = torch.softmax(att, dim=-1)
print("\nafter softmax - -inf became exactly 0, every row sums to 1:")
print(att[0, 0])
print(f"row sums: {att[0, 0].sum(dim=-1)}")
print("\nRead row 0: token 0 can only see itself, so 100% weight on itself.")
print("Row 3 sees tokens 0-3 and spreads its attention across them.")
print()

y = att @ vh
print(f"att @ v                              {tuple(y.shape)}   weighted average of values, per head")

y = y.transpose(1, 2)
print(f"  .transpose(1, 2)                   {tuple(y.shape)}   heads back beside channels")
print(f"  contiguous?                        {y.is_contiguous()}  <- .view() below would fail")
y = y.contiguous().view(B, T, C)
print(f"  .contiguous().view(B, T, C)        {tuple(y.shape)}   heads concatenated - back to the stream")
print()
print(f"Started {(B, T, C)}, ended {tuple(y.shape)}. Attention never changes the")
print("shape of the residual stream. It only changes what is written there.")
