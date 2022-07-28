import jax.random as random
import flax.linen as nn

input_size = 5

# Luckily this changing of input size causes an exception.
x = random.randint(random.PRNGKey(0), (input_size,), 0, 2)

perceptron = nn.Dense(4)
params = perceptron.init(random.PRNGKey(0), x)
print(perceptron.apply(params, x))
x = random.randint(random.PRNGKey(0), (input_size + 2,), 0, 2)
print(perceptron.apply(params, x))
