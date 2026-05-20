"""
Protocol module for SpecExtend Edge-Cloud communication.
Generated from draft.proto.
"""

from .draft_pb2 import DraftRequest, VerifyResult
from .draft_pb2_grpc import SpecExtendServiceStub, SpecExtendServiceServicer, add_SpecExtendServiceServicer_to_server

__all__ = [
    "DraftRequest",
    "VerifyResult",
    "SpecExtendServiceStub",
    "SpecExtendServiceServicer",
    "add_SpecExtendServiceServicer_to_server",
]
