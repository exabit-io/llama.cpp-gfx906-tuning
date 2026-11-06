# gfx906-rccl-topology

Writes the RCCL topology file for the XGMI ring of gfx906 GPUs from sysfs on every boot, and points every RCCL
process at it through `/etc/rccl.conf`. Built for the Mac Pro 7,1 with two Radeon Pro Vega II Duo modules and the
bridge: the firmware's XGMI hop table swaps the two bridge links, so RCCL's own detection builds rings on links that do
not exist. Measured on four dies (rccl-tests `all_reduce_perf`, 1 GiB, bus bandwidth):

| Topology | GB/s |
|---|---|
| generated (this package) | 34.1 |
| firmware's links | 24.8 |
| none (RCCL's own detection) | 24.6 |

The file is derived from the PCI tree and KFD's hives, never from fixed bus numbers, so adding cards or PCIe switch
pools changing needs no edit. Other gfx906 layouts get the firmware's links; other GPU families get no file. Details:
`man gfx906-rccl-topology`.

Files:

- `/usr/sbin/gfx906-rccl-topology`: the generator (Python 3, standard library only).
- `gfx906-rccl-topology.service`: runs it at boot after `resize-amdgpu-bars.service`, writes
  `/var/cache/gfx906-rccl-topology/topology.xml`.
- `/etc/rccl.conf` (conffile): `NCCL_TOPO_FILE=/var/cache/gfx906-rccl-topology/topology.xml`. Point it elsewhere for a
  hand-made topology; local edits survive upgrades.

Removing the package deletes the generated file (it names this boot's PCI addresses); RCCL then reports the missing
file and detects the topology itself. Purging also removes `/etc/rccl.conf`.

Build: `dpkg-buildpackage -b -us -uc` in this directory (debhelper 13, scdoc). Published as a signed apt repository in
the GitHub releases of exabit-io/resize-amdgpu-bars (rolling channel `gfx906-rccl-topology`).
