"""Exact in-context deletion for Kimi Linear (KDA + MLA hybrid) under MLX.

Companion to ``gemma_sv``. Where the Gemma work deletes from a grafted
support-vector memory by decrement, this package deletes from a hybrid
linear-attention model by restoring checkpointed recurrent state and replaying
the surviving suffix.
"""
