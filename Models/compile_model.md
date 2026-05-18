# Compile Model
[Docs](https://nequip.readthedocs.io/en/latest/guide/getting-started/workflow.html#compilation)

Compile on the target machine/GPU. AOTInductor needs GCC 8+ (`gcc --version`).

## AOTInductor (Recommended, PyTorch 2.6+)
```bash
nequip-compile my_model.nequip.zip compiled_model.nequip.pt2 \
  --device [cpu|cuda] \
  --mode aotinductor \
  --target [ase|pair_allegro]
```
- `pair_allegro`: LAMMPS MD
- `ase`: ASE scripts

## TorchScript (Legacy, PyTorch < 2.10)
```bash
nequip-compile my_model.nequip.zip compiled_model.nequip.pth \
  --device [cpu|cuda] \
  --mode torchscript
```

## Allegro Speedup
Append this flag to the command:
```bash
--modifiers enable_TritonContracter
```
