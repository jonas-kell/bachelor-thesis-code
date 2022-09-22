import traceback
from main import execute_computation
from slack.message import post_message_to_slack

for ansatz in ["single-real", "split-complex", "two-real"]:
    for depth, embed_dim in [(1, 16), (1, 8), (2, 8), (3, 4), (3, 8), (4, 4), (5, 4)]:
        for heads in [1, 2]:

            message = f"ansatz: {ansatz}, depth: {depth}, embed_dim: {embed_dim}, heads: {heads}"

            if (ansatz, depth, embed_dim, heads) in [
                ("single-real", 1, 16, 1),
                ("single-real", 5, 4, 1),
                ("single-real", 3, 4, 1),
                ("single-real", 3, 8, 1),
                ("split-complex", 5, 4, 1),
                ("split-complex", 3, 4, 1),
                ("split-complex", 3, 8, 1),
            ]:
                post_message_to_slack("skipped: " + message)
                print("skipped: " + message)
                continue  # skips one calculation (if already done or known to cause crashes)

            print("doing: " + message)
            try:
                execute_computation(
                    lattice_random_swaps=-1,
                    lattice_periodic=True,
                    lattice_size=4,
                    lattice_shape="trigonal_square",
                    ansatz=ansatz,
                    depth=depth,
                    embed_dim=embed_dim,
                    mlp_ratio=4,
                    num_heads=heads,
                    hamiltonian_J_parameter=-1,
                    hamiltonian_h_parameter=-0.7,
                    model_name="GPF-NNN",
                    early_abort_var=0.001,
                    n_steps=480,
                    n_samples=1000,
                )
            except Exception as exc:
                print(traceback.format_exc())

            post_message_to_slack("did: " + message)
            print("did: " + message)

post_message_to_slack("Have finished")
