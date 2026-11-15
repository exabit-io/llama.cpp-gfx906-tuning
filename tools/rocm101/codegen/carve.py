#!/usr/bin/env python3
"""carve.py OBJ OUT -- write the gfx906 code object (an ELF for EM_AMDGPU, e_machine 224) embedded in a HIP object file.
The object is a host ELF whose offload section holds the device ELF (ROCm 10.0: .hip_fatbin bundle; 10.1: .llvm.offloading)."""
import struct, sys
b = open(sys.argv[1], 'rb').read()
i, n = 0, 0
while True:
    i = b.find(b'\x7fELF', i + 1)
    if i < 0:
        break
    if b[i + 4] != 2 or struct.unpack_from('<H', b, i + 18)[0] != 224:
        continue
    shoff, = struct.unpack_from('<Q', b, i + 40)
    shentsize, shnum = struct.unpack_from('<HH', b, i + 58)
    end = shoff + shentsize * shnum
    for k in range(shnum):                                    # section contents can lie past the header table
        off, size = struct.unpack_from('<QQ', b, i + shoff + k * shentsize + 24)
        typ, = struct.unpack_from('<I', b, i + shoff + k * shentsize + 4)
        if typ != 8:
            end = max(end, off + size)
    open('%s.%d' % (sys.argv[2], n) if n else sys.argv[2], 'wb').write(b[i:i + end])
    n += 1
print(n, "device ELF(s)")
