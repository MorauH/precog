"""
Lock-free single-writer/single-reader ring buffer for experience transfer.

Backed by `multiprocessing.shared_memory.SharedMemory` with atomic head/tail
counters stored in the shared buffer header.

Writer (level process):
   1. Serialize snapshot to bytes (pickle)
   2. Write to slot[head], set length
   3. Advance head

Reader (learner process):
   1. Read slots from tail..head, collecting non-empty ones
   2. Deserialize and return list of snapshots
   3. Advance tail

No locks on the fast path — single-writer/single-reader is guaranteed by
the process-level architecture.
"""

from __future__ import annotations

import pickle
from multiprocessing.shared_memory import SharedMemory
from typing import Any, List, Optional

_HEADER_BYTES = 8  # 4 bytes length + 4 bytes reserved
_NUM_SLOTS_DEFAULT = 256
_SLOT_SIZE_DEFAULT = 65536  # 64KB per slot — covers typical PerLevelSnapshot


class ShmRingBuffer:
    """Ring buffer for experience snapshots, backed by shared memory.

    Args:
        name: Unique channel name.
        num_slots: Number of ring buffer slots.
        slot_size: Max serialized size per snapshot (bytes).
    """

    def __init__(
        self,
        name: str,
        num_slots: int = _NUM_SLOTS_DEFAULT,
        slot_size: int = _SLOT_SIZE_DEFAULT,
    ):
        self._name = name
        self._num_slots = num_slots
        self._slot_data_bytes = slot_size
        self._slot_total_bytes = _HEADER_BYTES + self._slot_data_bytes
        self._total_bytes = 4 + 4 + num_slots * self._slot_total_bytes
        # layout: [head (4B)] [tail (4B)] [slot_0_header+data] ... [slot_N_header+data]

        shm_name = f"shm_ring_{name}"
        try:
            self._shm = SharedMemory(name=shm_name, create=True, size=self._total_bytes)
        except FileExistsError:
            self._shm = SharedMemory(name=shm_name)
        self._buf = self._shm.buf
        self._head_ptr = 0
        self._tail_ptr = 4
        self._slot_base = 8

        import struct

        self._struct = struct

        self._closed = False

    @classmethod
    def attach(
        cls,
        name: str,
        num_slots: int = _NUM_SLOTS_DEFAULT,
        slot_size: int = _SLOT_SIZE_DEFAULT,
    ) -> ShmRingBuffer:
        """Attach to an existing ring buffer from another process."""
        total_bytes = 8 + num_slots * (_HEADER_BYTES + slot_size)
        rb = cls.__new__(cls)
        rb._name = name
        rb._num_slots = num_slots
        rb._slot_data_bytes = slot_size
        rb._slot_total_bytes = _HEADER_BYTES + slot_size
        rb._total_bytes = total_bytes
        rb._shm = SharedMemory(name=f"shm_ring_{name}")
        rb._buf = rb._shm.buf
        rb._head_ptr = 0
        rb._tail_ptr = 4
        rb._slot_base = 8
        import struct

        rb._struct = struct
        rb._closed = False
        return rb

    @property
    def name(self) -> str:
        return self._name

    def _read_u32(self, offset: int) -> int:
        return self._struct.unpack_from("<I", self._buf, offset)[0]

    def _write_u32(self, offset: int, value: int) -> None:
        self._struct.pack_into("<I", self._buf, offset, value)

    def _slot_offset(self, index: int) -> int:
        return self._slot_base + index * self._slot_total_bytes

    def write(self, obj: Any) -> bool:
        """Serialize and write an object to the next slot.

        Args:
            obj: Object to write (typically PerLevelSnapshot).

        Returns:
            True if written, False if buffer is full (reader hasn't drained).
        """
        data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
        if len(data) > self._slot_data_bytes:
            return False

        head = self._read_u32(self._head_ptr)
        tail = self._read_u32(self._tail_ptr)
        next_head = (head + 1) % self._num_slots
        if next_head == tail:
            return False

        slot_off = self._slot_offset(head)
        self._write_u32(slot_off, len(data))
        slot_data_off = slot_off + _HEADER_BYTES
        self._buf[slot_data_off : slot_data_off + len(data)] = data
        self._write_u32(self._head_ptr, next_head)
        return True

    def read_all(self, max_count: int = 0) -> List[Any]:
        """Drain all available objects from the buffer.

        Args:
            max_count: Max items to read (0 = unlimited).

        Returns:
            List of deserialized objects.
        """
        head = self._read_u32(self._head_ptr)
        tail = self._read_u32(self._tail_ptr)

        if tail == head:
            return []

        results: List[Any] = []
        count = 0

        while tail != head:
            slot_off = self._slot_offset(tail)
            length = self._read_u32(slot_off)
            if length == 0:
                tail = (tail + 1) % self._num_slots
                continue
            slot_data_off = slot_off + _HEADER_BYTES
            data = bytes(self._buf[slot_data_off : slot_data_off + length])
            obj = pickle.loads(data)
            results.append(obj)
            count += 1
            if max_count > 0 and count >= max_count:
                break
            tail = (tail + 1) % self._num_slots

        self._write_u32(self._tail_ptr, tail)
        return results

    def __getstate__(self):
        return {
            "name": self._name,
            "num_slots": self._num_slots,
            "slot_size": self._slot_data_bytes,
        }

    def __setstate__(self, state):
        rb = ShmRingBuffer.attach(**state)
        self._name = rb._name
        self._num_slots = rb._num_slots
        self._slot_data_bytes = rb._slot_data_bytes
        self._slot_total_bytes = rb._slot_total_bytes
        self._total_bytes = rb._total_bytes
        self._shm = rb._shm
        self._buf = rb._buf
        self._head_ptr = rb._head_ptr
        self._tail_ptr = rb._tail_ptr
        self._slot_base = rb._slot_base
        self._struct = rb._struct
        self._closed = rb._closed

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._shm.close()
            self._shm.unlink()
        except Exception:
            pass
