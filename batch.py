import traceback
from main import execute_computation
from slack.message import post_message_to_slack

try:
    execute_computation(
        lattice_random_swaps=-1,
        lattice_periodic=True,
        lattice_size=3,
        lattice_shape="trigonal_hexagonal",
        ansatz="single-real",
        depth=1,
        embed_dim=16,
        mlp_ratio=2,
        num_heads=1,
        hamiltonian_J_parameter=-1,
        hamiltonian_h_parameter=-0.7,
        model_name="CNN",
        head="act-fun",
        early_abort_var=-1,
        n_steps=800,
        n_samples=1000,
    )
except Exception as exc:
    print(traceback.format_exc())

post_message_to_slack("1/9")

try:
    execute_computation(
        lattice_random_swaps=-1,
        lattice_periodic=True,
        lattice_size=3,
        lattice_shape="trigonal_hexagonal",
        ansatz="single-real",
        depth=1,
        embed_dim=16,
        mlp_ratio=2,
        num_heads=1,
        hamiltonian_J_parameter=-1,
        hamiltonian_h_parameter=-0.7,
        model_name="RBM",
        head="act-fun",
        early_abort_var=-1,
        n_steps=800,
        n_samples=1000,
    )
except Exception as exc:
    print(traceback.format_exc())

post_message_to_slack("2/9")

try:
    execute_computation(
        lattice_random_swaps=-1,
        lattice_periodic=True,
        lattice_size=3,
        lattice_shape="trigonal_hexagonal",
        ansatz="single-real",
        depth=2,
        embed_dim=7,
        mlp_ratio=2,
        num_heads=1,
        hamiltonian_J_parameter=-1,
        hamiltonian_h_parameter=-0.7,
        model_name="GPF-NN",
        head="act-fun",
        early_abort_var=-1,
        n_steps=800,
        n_samples=1000,
    )
except Exception as exc:
    print(traceback.format_exc())

post_message_to_slack("3/9")

try:
    execute_computation(
        lattice_random_swaps=-1,
        lattice_periodic=True,
        lattice_size=3,
        lattice_shape="trigonal_hexagonal",
        ansatz="single-real",
        depth=2,
        embed_dim=7,
        mlp_ratio=2,
        num_heads=1,
        hamiltonian_J_parameter=-1,
        hamiltonian_h_parameter=-0.7,
        model_name="GPF-NNN",
        head="act-fun",
        early_abort_var=-1,
        n_steps=800,
        n_samples=1000,
    )
except Exception as exc:
    print(traceback.format_exc())

post_message_to_slack("4/9")

try:
    execute_computation(
        lattice_random_swaps=-1,
        lattice_periodic=True,
        lattice_size=3,
        lattice_shape="trigonal_hexagonal",
        ansatz="single-real",
        depth=2,
        embed_dim=7,
        mlp_ratio=2,
        num_heads=1,
        hamiltonian_J_parameter=-1,
        hamiltonian_h_parameter=-0.7,
        model_name="SGDCF-NN",
        head="act-fun",
        early_abort_var=-1,
        n_steps=800,
        n_samples=1000,
    )
except Exception as exc:
    print(traceback.format_exc())

post_message_to_slack("5/9")

try:
    execute_computation(
        lattice_random_swaps=-1,
        lattice_periodic=True,
        lattice_size=3,
        lattice_shape="trigonal_hexagonal",
        ansatz="single-real",
        depth=2,
        embed_dim=7,
        mlp_ratio=2,
        num_heads=1,
        hamiltonian_J_parameter=-1,
        hamiltonian_h_parameter=-0.7,
        model_name="SGDCF-NNN",
        head="act-fun",
        early_abort_var=-1,
        n_steps=800,
        n_samples=1000,
    )
except Exception as exc:
    print(traceback.format_exc())

post_message_to_slack("6/9")

try:
    execute_computation(
        lattice_random_swaps=-1,
        lattice_periodic=True,
        lattice_size=3,
        lattice_shape="trigonal_hexagonal",
        ansatz="single-real",
        depth=2,
        embed_dim=5,
        mlp_ratio=2,
        num_heads=1,
        hamiltonian_J_parameter=-1,
        hamiltonian_h_parameter=-0.7,
        model_name="TF",
        head="act-fun",
        early_abort_var=-1,
        n_steps=800,
        n_samples=1000,
    )
except Exception as exc:
    print(traceback.format_exc())

post_message_to_slack("7/9")

try:
    execute_computation(
        lattice_random_swaps=-1,
        lattice_periodic=True,
        lattice_size=3,
        lattice_shape="trigonal_hexagonal",
        ansatz="single-real",
        depth=2,
        embed_dim=5,
        mlp_ratio=2,
        num_heads=1,
        hamiltonian_J_parameter=-1,
        hamiltonian_h_parameter=-0.7,
        model_name="GTF-NN",
        head="act-fun",
        early_abort_var=-1,
        n_steps=800,
        n_samples=1000,
    )
except Exception as exc:
    print(traceback.format_exc())

post_message_to_slack("8/9")

try:
    execute_computation(
        lattice_random_swaps=-1,
        lattice_periodic=True,
        lattice_size=3,
        lattice_shape="trigonal_hexagonal",
        ansatz="single-real",
        depth=2,
        embed_dim=5,
        mlp_ratio=2,
        num_heads=1,
        hamiltonian_J_parameter=-1,
        hamiltonian_h_parameter=-0.7,
        model_name="GTF-NNN",
        head="act-fun",
        early_abort_var=-1,
        n_steps=800,
        n_samples=1000,
    )
except Exception as exc:
    print(traceback.format_exc())

post_message_to_slack("9/9")

post_message_to_slack("Have finished")
