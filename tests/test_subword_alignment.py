"""
THE ALIGNMENT TEST. Run before any subword training.

    python tests/test_subword_alignment.py

Labels are one per WORD. Subword pieces are smaller than words. If the piece
tensor ever has a different number of word-rows than the label tensor, labels
shift against words. The model still trains, the loss still falls, and the score
is quietly terrible. Nothing crashes. This test is the only thing that catches
it early.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch

from dataset import collate, SOLDTokenDataset
from model import SubwordEncoder, BiLSTMTagger


class FakeSP:
    """Deterministic stand-in for SentencePiece: splits a word every 3 chars."""
    def __init__(self, n_pieces=500):
        self.n = n_pieces

    def encode(self, word, out_type=int):
        chunks = [word[i:i + 3] for i in range(0, len(word), 3)] or [word]
        if out_type is str:
            return chunks
        return [(abs(hash(c)) % (self.n - 2)) + 2 for c in chunks]

    def unk_id(self):
        return 1

    def get_piece_size(self):
        return self.n


def fake_df(rows):
    import pandas as pd
    return pd.DataFrame([
        {"token_list": t, "rationales": r, "label": lab}
        for t, r, lab in rows
    ])


ok = lambda m: print(f"  {m:<58} OK")

# ---------------------------------------------------------------- 1
print("\n1. Dataset produces one piece-list per word")
sp = FakeSP()
df = fake_df([
    (["abcdef", "gh", "ijklmnopq"], [0, 1, 0], "OFF"),
    (["a"], [0], "NOT"),
    (["ab", "cd", "ef", "gh", "ij"], [0, 0, 1, 1, 0], "OFF"),
])
ds = SOLDTokenDataset(df, {"<PAD>": 0, "<UNK>": 1}, sp=sp)
for i in range(len(ds)):
    ids, lab, _, pcs = ds[i]
    assert len(pcs) == len(ids) == len(lab), (i, len(pcs), len(ids), len(lab))
ok("every tweet: len(pieces) == len(words) == len(labels)")

# ---------------------------------------------------------------- 2
print("\n2. Collate preserves the invariant after padding")
batch = [ds[i] for i in range(len(ds))]
w, lab, mask, lens, sent, pid, plen = collate(batch)
assert pid.shape[0] == w.shape[0], "batch size mismatch"
assert pid.shape[1] == w.shape[1], f"word axis: pieces {pid.shape[1]} vs words {w.shape[1]}"
assert plen.shape == w.shape
ok(f"piece block {tuple(pid.shape)} matches word block {tuple(w.shape)}")

# padded word positions must have zero pieces
for i in range(w.shape[0]):
    n = int(lens[i])
    assert (plen[i, :n] > 0).all(), "a real word has zero pieces"
    assert (plen[i, n:] == 0).all(), "a padded word has pieces"
ok("real words have >=1 piece; padded words have 0")

# ---------------------------------------------------------------- 3
print("\n3. Encoder returns exactly one vector per word")
enc = SubwordEncoder(n_pieces=500, piece_dim=16, out_dim=32, pooling="bilstm")
enc.eval()   # dropout off: these checks must be deterministic
out = enc(pid, plen)
assert out.shape == (w.shape[0], w.shape[1], 32), out.shape
ok(f"bilstm pooling: {tuple(pid.shape)} -> {tuple(out.shape)}")

enc_m = SubwordEncoder(n_pieces=500, piece_dim=16, out_dim=32, pooling="mean")
enc_m.eval()
out_m = enc_m(pid, plen)
assert out_m.shape == out.shape
ok(f"mean pooling:   {tuple(pid.shape)} -> {tuple(out_m.shape)}")

# ---------------------------------------------------------------- 4
print("\n4. Padded word positions produce all-zero vectors")
for i in range(w.shape[0]):
    n = int(lens[i])
    if n < w.shape[1]:
        assert torch.allclose(out[i, n:], torch.zeros_like(out[i, n:])), \
            "padded position is not zeroed"
ok("no signal leaks from padding into the word representation")

# ---------------------------------------------------------------- 5
print("\n5. A word's vector depends only on ITS OWN pieces")
# Change the pieces of word 0 only; word 1's vector must not move.
pid2 = pid.clone()
pid2[0, 0, 0] = (int(pid2[0, 0, 0]) + 7) % 400 + 2
out2 = enc(pid2, plen)
assert not torch.allclose(out[0, 0], out2[0, 0]), "changing pieces did nothing"
assert torch.allclose(out[0, 1], out2[0, 1], atol=1e-6), \
    "changing word 0's pieces moved word 1 - the words are bleeding into each other"
ok("pieces do not leak across word boundaries")

# ---------------------------------------------------------------- 6
print("\n6. Full model: emissions have one row per word")
matrix = np.random.RandomState(0).normal(0, 0.1, (50, 300)).astype(np.float32)
for pooling in ("bilstm", "mean"):
    m = BiLSTMTagger(matrix, hidden_size=16, use_crf=False,
                     n_pieces=500, piece_dim=16, subword_dim=32,
                     subword_pooling=pooling)
    m.eval()
    em = m.emissions(w, lens, pid, plen)
    assert em.shape == (w.shape[0], w.shape[1], 2), em.shape
    ok(f"{pooling:<6} emissions {tuple(em.shape)}; lstm input dim "
       f"{m.count_parameters()['lstm_input_dim']}")

# ---------------------------------------------------------------- 7
print("\n7. Word channel alone still works (Phase 1 path unbroken)")
m_word = BiLSTMTagger(matrix, hidden_size=16, use_crf=False)
m_word.eval()
em = m_word.emissions(w, lens)
assert em.shape == (w.shape[0], w.shape[1], 2)
ok("no subword args needed when n_pieces=0")

# ---------------------------------------------------------------- 8
print("\n8. Subword channel alone works (ablation row)")
m_sub = BiLSTMTagger(matrix, hidden_size=16, use_crf=False, n_pieces=500,
                     subword_dim=32, use_word_channel=False)
m_sub.eval()
em = m_sub.emissions(w, lens, pid, plen)
assert em.shape == (w.shape[0], w.shape[1], 2)
assert m_sub.count_parameters()["lstm_input_dim"] == 32
ok("word channel can be switched off for the ablation")

# ---------------------------------------------------------------- 9
print("\n9. A deliberately misaligned encoder is CAUGHT, not silently accepted")
class BrokenEncoder(torch.nn.Module):
    out_dim = 32
    def forward(self, piece_ids, piece_lens):
        B, W, P = piece_ids.shape
        return torch.zeros(B, W - 1, 32)     # one row short
m_bad = BiLSTMTagger(matrix, hidden_size=16, use_crf=False, n_pieces=500, subword_dim=32)
m_bad.subword = BrokenEncoder()
try:
    m_bad.emissions(w, lens, pid, plen)
    raise SystemExit("FAIL: misalignment was NOT caught")
except RuntimeError as e:
    assert "ALIGNMENT BROKEN" in str(e)
ok("RuntimeError raised with a clear message")

# ---------------------------------------------------------------- 10
print("\n10. Missing piece ids raise a clear error, not a crash")
try:
    BiLSTMTagger(matrix, hidden_size=16, use_crf=False,
                 n_pieces=500, subword_dim=32).emissions(w, lens)
    raise SystemExit("FAIL: missing pieces not caught")
except ValueError as e:
    assert "no piece ids" in str(e)
ok("ValueError tells you to pass sp=... to make_loader")

print("\nAll alignment tests passed.\n")
