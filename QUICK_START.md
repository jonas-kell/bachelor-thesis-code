### Adapt folder constant in `main.py`:

```python
# local folder constants
tensorboard_folder_path = "/<path>/<to>/<tensorboard>/<folder>/"
```

### Visualize lattices

Set the desired lattices on the bottom of `/structures/visualize.ps`:
(Uncomment and set parameters)

```python
if __name__ == "__main__":
    # draw_linear_lattice(6, False, 0, False)
    draw_cubic_lattice(4, False, 0, False)
    # draw_trigonal_square_lattice(4, False, 0, False)
    # draw_trigonal_diamond_lattice(4, False, 0, False)
    # draw_trigonal_hexagonal_lattice(4, False, 0, False)
    # draw_hexagonal_lattice(3, True, 0, False)
```

```cmd
cd visualize
python3 visualize.py
```

### Perform ground state search

```cmd
python3 main.py
```

Additional parameters that modify the training behavior can be specified:

```cmd
python3 main.py <<argument>>=<<value>>
```

At the moment the following parameters are available:

```
"<<argument>>=<<example-value>>"

"n_steps=1000"
"n_samples=1000"
"lattice_shape=linear"
"lattice_size=6"
"lattice_periodic=True"
"lattice_random_swaps=0"
"model_name=GC-NNN"
"hamiltonian_J_parameter=-1.0"
"hamiltonian_h_parameter=-0.7"
"num_chains=100"
"thermalization_sweeps=25"
"nqs_batch_size=1000"
"depth=3"
"embed_dim=2"
"num_heads=1"
"mlp_ratio=2"
"ansatz=single-complex"
"early_abort_var=-1.0"
```
