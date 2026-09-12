# Glossary

Every term used in this repo, in plain language. No prior linear algebra
assumed. Read it once, refer back as needed.

## The only maths you actually need

**Dot product.** Two lists of the same length. Multiply position by position,
add up the results. One number out.

```
[2, 0, 1] · [3, 4, 1]  =  2*3 + 0*4 + 1*1  =  7
```

Why it matters: the result is large when the two lists point the same
direction and small when they don't. It is a **similarity score**. That is the
entire reason attention uses it.

**Matrix multiplication** (the `@` operator). Doing many dot products at once.
If A is `(4, 3)` and B is `(3, 5)`, then `A @ B` is `(4, 5)`, and the cell at
row *i*, column *j* is the dot product of A's row *i* with B's column *j*.

The inner numbers must match - `(4, 3) @ (3, 5)` works because both are 3.
This is what almost every shape error in ML is about. Roughly 90% of the
debugging you will do is making the inner dimensions agree.

**Transpose.** Flip rows and columns. `(3, 5)` becomes `(5, 3)`. Usually done
so the inner dimensions line up for a matmul. `k.transpose(-2, -1)` means
"swap the last two dimensions of k," leaving any leading dimensions alone.

That's it. Dot product, matmul, transpose. Everything else is arrangement.

## Shapes

**Tensor.** An array of numbers with any number of dimensions. A single number
is 0-D, a list is 1-D, a table is 2-D, a stack of tables is 3-D. "Tensor" just
means "don't care how many dimensions, it's numbers in a box."

**Shape.** The size of each dimension, as a tuple. `(2, 4, 6)` means 2 of
(4 rows of 6 numbers). Reading shapes fluently is the single most useful skill
in this whole area.

**view / reshape.** Reinterpret the same numbers with a different shape. Free -
no data moves, only the bookkeeping about how to walk memory changes. The total
count must stay the same: `(2, 4, 6)` holds 48 numbers and can become
`(2, 4, 2, 3)`, also 48.

**contiguous.** Whether a tensor's numbers sit in memory in the order its shape
implies. `transpose` makes a tensor non-contiguous - it changes the walking
order without moving anything. `.view()` requires contiguous, which is why
`.contiguous().view(...)` appears after a transpose.

## The dimensions in this code

**B - batch.** How many independent sequences are processed at once. Purely a
hardware efficiency thing: a GPU running 64 sequences takes barely longer than
one. Nothing conceptual happens across the batch dimension; the sequences never
interact.

**T - time.** Position in the sequence. `T=256` means 256 tokens of context.
Also called `block_size` or context length.

**C - channels.** How many numbers represent each token. Also called `n_embd`,
embedding dimension, model dimension, or `d_model` - all the same thing. This
is the "width" of the model.

**H - heads.** How many independent attention lookups run in parallel.

**hs - head size.** `C // H`. Each head gets its own slice of the channels.

## Model terms

**Embedding.** A lookup table that turns a symbol into a vector. Token 41
becomes row 41 of a table. The rows are learned, so "meaning" ends up as a
position in space.

**Logits.** Raw, unnormalised scores straight out of the model, before they are
turned into probabilities. Can be any real number, positive or negative.

**Softmax.** Turns any list of numbers into probabilities: all positive, summing
to 1. Bigger inputs get bigger shares. `-inf` maps to exactly 0, which is why
masking uses it.

**Linear layer.** A learned matrix multiply, optionally plus a bias.
`Linear(6, 18)` holds a `(6, 18)` matrix and turns 6 numbers into 18. Applied
independently to every position.

**Attention.** Each position asks "what should I be looking at?" and gathers a
weighted average of information from other positions. Weights come from dot
products between queries and keys.

**Query / Key / Value.** Three vectors each position produces from its own
input. Query = what I'm looking for. Key = what I offer. Value = what I hand
over if chosen. Scores come from query-dot-key; the output is a weighted
average of values.

**Head.** One independent attention lookup. Multiple heads can specialise -
one tracking grammar, another long-range references. Their outputs are
concatenated.

**Causal mask.** Prevents a position from attending to later positions. Without
it the model can see the answer it is being asked to predict.

**Residual stream.** The `(B, T, C)` tensor flowing down the model. Each block
*adds* to it rather than replacing it. That additive path is what makes deep
networks trainable.

**Layer norm.** Rescales a vector to mean 0 and variance 1. Keeps numbers in a
sane range so training doesn't destabilise.

**Dropout.** During training only, randomly zero some fraction of values. Stops
the model leaning too hard on any single pathway. Disabled at inference.

## Training terms

**Loss.** One number saying how wrong the model currently is. Lower is better.
Training is the process of making it smaller.

**Cross-entropy.** The loss used for predicting one option out of many. Equal to
`ln(vocab_size)` when the model is guessing uniformly - about 4.17 for 65
characters, which is why `train.py` prints that as a sanity check.

**Gradient.** For each parameter, which direction to nudge it to reduce the
loss. Computed by `loss.backward()`.

**Gradient descent.** Repeatedly nudge every parameter slightly in the
direction its gradient points. That is all training is.

**Optimizer (AdamW).** The thing that performs the nudge. Adam adapts the step
size per parameter based on recent gradient history; the W is a particular way
of applying weight decay.

**Learning rate.** How big each nudge is. The most important single knob in
training.

**Epoch / iteration.** An iteration is one batch, one update. An epoch is one
full pass over the data. This code counts iterations.

**Parameters.** The numbers that get learned. ~10.7M here. Weights and biases.

## Precision

**fp32 / fp16 / bf16 / tf32.** How many bits represent each number. fp32 is 32
bits, fp16 and bf16 are 16. bf16 keeps fp32's range with less precision, which
turns out to be the right trade for deep learning - and on this card it is also
the fastest path. See README for measured numbers.

**Tensor cores.** Dedicated silicon that does small matrix multiplies very
fast, but only in reduced precision. Why bf16 is roughly 5x faster than fp32
here.

**Quantisation.** Storing weights in even fewer bits (8, 4) to fit larger
models in limited VRAM. The main lever on a 12GB card.
