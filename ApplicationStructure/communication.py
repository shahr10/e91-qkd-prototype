"""
================================================================================
NETWORKING FUNCTIONS FOR E91 QKD
================================================================================

UDP-based two-way communication for quantum-secured messaging.

These functions enable live computer-to-computer quantum-secured messaging
using UDP for simplicity. Production systems should use TCP with proper
authentication and error handling.

Author: Tyler Barr
Version: 7.0.0 Modular
Date: 2025

================================================================================
"""

import socket
import logging
from typing import Optional, Tuple


# ============================================================================
# UDP COMMUNICATION FUNCTIONS
# ============================================================================

def send_udp_message(host: str, port: int, payload: bytes, timeout: float = 2.0) -> None:
    """
    Send a UDP payload to a peer (best-effort, no ACK).

    Args:
        host: Target IP address or hostname
        port: Target UDP port number
        payload: Encrypted message bytes to send
        timeout: Socket timeout in seconds
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.settimeout(timeout)
            s.sendto(payload, (host, int(port)))
    except Exception as exc:
        logging.warning(f"UDP send failed to {host}:{port} - {exc}")


def receive_udp_message(port: int, timeout: float = 2.0) -> Optional[Tuple[bytes, Tuple[str, int]]]:
    """
    Receive a single UDP payload (if any) on the given port.

    Args:
        port: Local UDP port to listen on
        timeout: How long to wait for a message (seconds)

    Returns:
        Tuple of (data, sender_address) if message received, None on timeout
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.bind(("", int(port)))
            s.settimeout(timeout)
            data, addr = s.recvfrom(65535)
            return data, addr
    except socket.timeout:
        return None
    except Exception as exc:
        logging.warning(f"UDP receive failed on port {port} - {exc}")
        return None


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'send_udp_message',
    'receive_udp_message',
]
