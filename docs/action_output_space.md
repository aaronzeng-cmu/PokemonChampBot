# Action Output Space (BC Transformer)

## Decision: single integer combo index

**Date:** 2026-06-08

VGC doubles requires simultaneous actions for Slot 1 (`pXa`) and Slot 2 (`pXb`). We encode the joint action as **one integer**:

```
combo_index = slot0_action * 107 + slot1_action
```

| Constant | Value |
|----------|-------|
| `ACTION_SIZE` | 107 (poke-env Gen 9 doubles per-slot indices) |
| `COMBO_VOCAB_SIZE` | 11,449 |

### Per-slot index semantics (poke-env)

| Range | Meaning |
|-------|---------|
| 0 | Pass |
| 1–6 | Switch to team slot |
| 7–106 | Move (+ target offset), with mega/tera/dmax offsets |

### Why not dual labels?

Two separate CE heads were prototyped earlier but the project spec prefers a **single master combo index** for simpler training targets and direct alignment with `enumerate_legal_combos` at inference (decode pair → find matching legal combo or mask).

### Inference masking

1. Model outputs logits over `COMBO_VOCAB_SIZE` (or dual-head decode → combo_index).
2. Decode to `(slot0, slot1)`.
3. Validate against `DoublesEnv.get_action_mask_individual` per slot.
4. If joint-illegal, fall back to next-best legal combo.

Logged copy: `logs/parser_sanity/action_space_decision.json`
