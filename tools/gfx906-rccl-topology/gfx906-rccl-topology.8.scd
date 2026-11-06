gfx906-rccl-topology(8)

# NAME

gfx906-rccl-topology - write the RCCL topology for the XGMI ring of gfx906 GPUs

# SYNOPSIS

*gfx906-rccl-topology* *generate* [*--output* _PATH_] [*--links* *rule*|*firmware*] [*--wait* _SECONDS_]

*gfx906-rccl-topology* *show* [*--links* *rule*|*firmware*]

*gfx906-rccl-topology* *--version*

# DESCRIPTION

RCCL reads the XGMI links between AMD GPUs from the hop table the firmware
reports. On the Mac Pro 7,1 with two Radeon Pro Vega II Duo modules and the
bridge between them, that table swaps the two bridge links, so RCCL builds its
rings on links that do not exist and all-reduce runs at about 25 instead of
34 GB/s. *gfx906-rccl-topology* writes an RCCL/NCCL topology file
(*NCCL_TOPO_FILE*) with the physical links instead, derived from sysfs on
every run, so PCI renumbering (cards added or moved, PCIe switch pools
changing) needs no manual edit.

For each XGMI hive (KFD _hive_id_):

- module: the outermost PCIe switch above a GPU (the first upstream port
  below the root port);
- die: the downstream port of that switch the GPU hangs off, ordered by
  device number;
- links: the two dies of a module and, with two modules in one hive, the
  dies on the same switch port of each module.

A hive of another shape gets the firmware's one-hop XGMI links, with a
warning. Every link is checked against KFD's XGMI table. Only gfx906 GPUs are
supported: if KFD reports any other GPU, nothing is written.

# COMMANDS

*generate*
	Write the topology to _PATH_ (atomically, mode 0644). On any failure
	nothing is written and a file from an earlier run is renamed to
	_PATH_.stale, so RCCL never reads a topology that may name PCI addresses
	which no longer exist.

*show*
	Print the topology on standard output.

The links chosen are logged on standard error in both cases.

# OPTIONS

*--output* _PATH_
	Default: _/var/cache/gfx906-rccl-topology/topology.xml_.

*--links* *rule*|*firmware*
	*rule* (default): the physical ring described above. *firmware*: the
	firmware's one-hop XGMI links, for comparison.

*--wait* _SECONDS_
	Wait up to _SECONDS_ for KFD to report every AMD GPU that is bound to
	amdgpu or not bound to any driver yet. GPUs bound to another driver
	(vfio-pci) are not waited for. Default 0.

# FILES

_/var/cache/gfx906-rccl-topology/topology.xml_
	Written at every boot by *gfx906-rccl-topology.service*, which runs
	after *resize-amdgpu-bars.service*.

_/etc/rccl.conf_
	Sets *NCCL_TOPO_FILE* to the file above for every RCCL process. To use a
	hand-made topology, point that line at your own file. *NCCL_TOPO_FILE* in
	the environment overrides it.

# EXIT STATUS

*0*
	Written or shown, or no AMD GPUs at all (nothing to do).

*1*
	A GPU other than gfx906 in KFD, no GPUs in KFD, or links that are not
	XGMI peers in KFD. Nothing is written.

# EXAMPLES

Check what was written at boot:

	journalctl -b -u gfx906-rccl-topology.service

Regenerate after changing hardware without a reboot:

	systemctl restart gfx906-rccl-topology.service

# SEE ALSO

*systemctl*(1), *journalctl*(1), *resize-amdgpu-bars*(8)
