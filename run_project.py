from src.config import IMAGES_PATH, LAYER_SIZES
from src.model import initialize_model
from src.search_space import create_search_space
from src.astar import run_rover_simulation
from src.visualization import visualize_search_space, visualize_rover_path


model = initialize_model()

G, layers = create_search_space(
    LAYER_SIZES
)

print("Total nodes:", len(G.nodes))
print("Total edges:", len(G.edges))

visualize_search_space(
    G,
    layers
)

path, movement_history = run_rover_simulation(
    G,
    layers,
    IMAGES_PATH,
    model
)

visualize_rover_path(
    G,
    layers,
    path
)
