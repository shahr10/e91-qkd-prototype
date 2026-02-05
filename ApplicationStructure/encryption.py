"""
================================================================================
MESSAGE ENCRYPTION FOR E91 QKD
================================================================================

One-Time Pad (OTP) encryption using QKD-generated keys.
Demonstrates practical use of quantum keys for secure communication.

Author: Tyler Barr
Version: 7.0.0 Modular
Date: 2025

================================================================================
"""

import time
from typing import List
from .models import MessageTest


# ============================================================================
# BIT CONVERSION FUNCTIONS
# ============================================================================

def message_to_bits(message: str) -> List[int]:
    """
    Convert message to bits.

    Args:
        message: Plain text message

    Returns:
        List of bits (0 or 1)
    """
    bits = []
    for byte in message.encode('utf-8'):
        for i in range(8):
            bits.append((byte >> (7 - i)) & 1)
    return bits


def bits_to_message(bits: List[int]) -> str:
    """
    Convert bits to message.

    Args:
        bits: List of bits (0 or 1)

    Returns:
        Decoded string message
    """
    while len(bits) % 8 != 0:
        bits.append(0)
    byte_array = bytearray()
    for i in range(0, len(bits), 8):
        byte_val = sum((bits[i + j] << (7 - j)) for j in range(8) if i + j < len(bits))
        byte_array.append(byte_val)
    return byte_array.decode('utf-8', errors='ignore')


def xor_encrypt_decrypt(message_bits: List[int], key_bits: List[int]) -> List[int]:
    """
    XOR encryption/decryption (symmetric operation).

    Args:
        message_bits: Message bits to encrypt/decrypt
        key_bits: One-time pad key bits

    Returns:
        Encrypted/decrypted bits

    Raises:
        ValueError: If key is shorter than message
    """
    if len(key_bits) < len(message_bits):
        raise ValueError(f"Key too short: need {len(message_bits)}, have {len(key_bits)}")
    return [message_bits[i] ^ key_bits[i] for i in range(len(message_bits))]


# ============================================================================
# MESSAGE ENCRYPTION/DECRYPTION FUNCTIONS
# ============================================================================

def test_self_message(message: str, quantum_key: List[int]) -> MessageTest:
    """
    Self-test message encryption (loopback test).
    Encrypts and decrypts with the same key to verify the OTP mechanism.

    Args:
        message: Plain text message to encrypt
        quantum_key: QKD-generated key bits

    Returns:
        MessageTest result with encryption/decryption details
    """
    start_time = time.time()
    try:
        message_bits = message_to_bits(message)
        if len(quantum_key) < len(message_bits):
            return MessageTest("self_test", message, b"", "", [], len(quantum_key),
                             "One-Time Pad", False,
                             f"Key too short: need {len(message_bits)}, have {len(quantum_key)}",
                             time.time() - start_time)
        key_subset = quantum_key[:len(message_bits)]
        encrypted_bits = xor_encrypt_decrypt(message_bits, key_subset)
        encrypted_bytes = bytearray(sum((encrypted_bits[i + j] << (7 - j))
                                       for j in range(8))
                                   for i in range(0, len(encrypted_bits), 8))
        decrypted_bits = xor_encrypt_decrypt(encrypted_bits, key_subset)
        decrypted_message = bits_to_message(decrypted_bits)
        return MessageTest("self_test", message, bytes(encrypted_bytes), decrypted_message,
                         key_subset, len(key_subset), "One-Time Pad",
                         decrypted_message == message, None, time.time() - start_time)
    except Exception as e:
        return MessageTest("self_test", message, b"", "", [], 0, "One-Time Pad",
                         False, str(e), time.time() - start_time)


def test_two_party_message(message: str, alice_key: List[int], bob_key: List[int]) -> MessageTest:
    """
    Two-party message test (Alice encrypts, Bob decrypts).

    Args:
        message: Plain text message
        alice_key: Alice's QKD key bits
        bob_key: Bob's QKD key bits

    Returns:
        MessageTest result showing if keys match and message decrypts correctly
    """
    start_time = time.time()
    try:
        message_bits = message_to_bits(message)
        min_key_len = min(len(alice_key), len(bob_key))
        if min_key_len < len(message_bits):
            return MessageTest("two_party", message, b"", "", [], min_key_len,
                             "One-Time Pad", False,
                             f"Keys too short: need {len(message_bits)}, have {min_key_len}",
                             time.time() - start_time, alice_key, bob_key)
        alice_key_subset = alice_key[:len(message_bits)]
        bob_key_subset = bob_key[:len(message_bits)]
        mismatches = [i for i in range(len(message_bits))
                     if alice_key_subset[i] != bob_key_subset[i]]
        encrypted_bits = xor_encrypt_decrypt(message_bits, alice_key_subset)
        encrypted_bytes = bytearray(sum((encrypted_bits[i + j] << (7 - j))
                                       for j in range(8))
                                   for i in range(0, len(encrypted_bits), 8))
        decrypted_bits = xor_encrypt_decrypt(encrypted_bits, bob_key_subset)
        decrypted_message = bits_to_message(decrypted_bits)
        success = (decrypted_message == message)
        error_msg = (f"Key mismatch at {len(mismatches)} positions"
                    if not success and len(mismatches) > 0 else None)
        return MessageTest("two_party", message, bytes(encrypted_bytes), decrypted_message,
                         alice_key_subset, len(alice_key_subset), "One-Time Pad", success,
                         error_msg, time.time() - start_time, alice_key_subset,
                         bob_key_subset, mismatches)
    except Exception as e:
        return MessageTest("two_party", message, b"", "", [], 0, "One-Time Pad",
                         False, str(e), time.time() - start_time, alice_key, bob_key)


def listener_receive_message(encrypted_message: bytes, bob_key: List[int]) -> MessageTest:
    """
    Listener receives and decrypts message from network.

    Args:
        encrypted_message: Encrypted message bytes received
        bob_key: Bob's QKD key bits for decryption

    Returns:
        MessageTest result with decrypted message
    """
    start_time = time.time()
    try:
        encrypted_bits = [((byte >> (7 - i)) & 1) for byte in encrypted_message for i in range(8)]
        if len(bob_key) < len(encrypted_bits):
            return MessageTest("listener", "[Unknown]", encrypted_message, "", [], len(bob_key),
                             "One-Time Pad", False, f"Insufficient key", time.time() - start_time)
        key_subset = bob_key[:len(encrypted_bits)]
        decrypted_bits = xor_encrypt_decrypt(encrypted_bits, key_subset)
        decrypted_message = bits_to_message(decrypted_bits)
        return MessageTest("listener", "[Unknown]", encrypted_message, decrypted_message,
                         key_subset, len(key_subset), "One-Time Pad", True, None,
                         time.time() - start_time, bob_key=key_subset)
    except Exception as e:
        return MessageTest("listener", "[Unknown]", encrypted_message, "", [], 0,
                         "One-Time Pad", False, str(e), time.time() - start_time)


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'message_to_bits',
    'bits_to_message',
    'xor_encrypt_decrypt',
    'test_self_message',
    'test_two_party_message',
    'listener_receive_message',
]
