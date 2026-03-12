"""
PATENT ELEMENT G/G1: Hash-Chain Verification System
"""

import hashlib
import json


class HashChainVerification:
    """
    PATENT ELEMENT G/G1: Cryptographically hashed and maintained in a secure
    audit trail or ledger.

    True hash-chain implementation where each valuation record includes the
    hash of the previous record, creating a tamper-evident linked chain.
    Modifying any historical record breaks the chain.

    Chain structure:
    Block N: { data, prev_hash: hash(Block N-1), hash: SHA256(data + prev_hash) }
    """

    def __init__(self):
        self.chain = []
        self.token_registry = {}

    def _compute_hash(self, data_str):
        return hashlib.sha256(data_str.encode('utf-8')).hexdigest()

    def add_record(self, property_id, sensor_data, indicators, valuation, timestamp):
        """Add a new record to the hash chain."""
        prev_hash = self.chain[-1]["block_hash"] if self.chain else "0" * 64
        block_index = len(self.chain)

        record = {
            "block_index": block_index,
            "timestamp": timestamp,
            "property_id": property_id,
            "sensor_summary": {k: round(v, 4) if isinstance(v, float) else v
                               for k, v in sensor_data.items()},
            "indicators": {k: round(v, 4) if isinstance(v, float) else v
                           for k, v in indicators.items()},
            "rtpmv": round(valuation["rtpmv"], 2),
            "health_factor": round(valuation["health_factor"], 4),
            "prev_hash": prev_hash,
        }

        # Canonical JSON → SHA-256
        record_json = json.dumps(record, sort_keys=True)
        block_hash = self._compute_hash(record_json)

        block = {**record, "block_hash": block_hash}
        self.chain.append(block)

        return block

    def verify_chain_integrity(self):
        """Verify entire chain — returns (is_valid, break_index)."""
        for i in range(len(self.chain)):
            block = self.chain[i]
            if i == 0:
                expected_prev = "0" * 64
            else:
                expected_prev = self.chain[i - 1]["block_hash"]
            if block["prev_hash"] != expected_prev:
                return False, i

            record = {k: v for k, v in block.items() if k != "block_hash"}
            record_json = json.dumps(record, sort_keys=True)
            recomputed = self._compute_hash(record_json)
            if recomputed != block["block_hash"]:
                return False, i

        return True, -1

    def generate_token(self, rtpmv, health_factor, ci, property_id, timestamp):
        """Generate valuation token with chain reference."""
        chain_head = self.chain[-1]["block_hash"] if self.chain else "genesis"
        token = {
            "token_id": f"VT-{property_id}-{len(self.chain):06d}",
            "property_id": property_id,
            "rtpmv": round(rtpmv, 2),
            "health_factor": round(health_factor, 4),
            "confidence_index": round(ci, 4),
            "timestamp": timestamp,
            "chain_head_hash": chain_head,
            "chain_length": len(self.chain),
            "token_standard": "RTPV-2.0",
        }
        token_json = json.dumps(token, sort_keys=True)
        token["token_hash"] = self._compute_hash(token_json)
        self.token_registry[token["token_id"]] = token
        return token

    def get_recent_chain(self, n=10):
        return self.chain[-n:]
