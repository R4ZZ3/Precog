"""Learned next-token expert prefetch for MoE decode with offloaded experts.

The FreeToken integration is ``commitmoe_prefetch.freetoken.create_decode_hooks``, which
FreeToken's ``--expert-predictor`` loads. The other modules do not import FreeToken.
"""
