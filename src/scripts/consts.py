# This script defines constants used across various modules in the data annotation pipeline.

RND_STATE = 42

# Data dependent constants
DEFAULT_LABEL_COLUMN = "Serotype"
DEFAULT_MISSING_LABEL = "Non-typeable"
CONTIG_SEP = "#"

# 2D visualization
DEFAULT_DOWNSAMPLE_SIZE = 1000

# Inference, training, and classification
DEFAULT_SEP = "|"
DEFAULT_OUTPUT_DIM = 128
DEFAULT_BATCH_SIZE = 32
DEFAULT_NUM_LAYERS = 1
DEFAULT_NHEAD = 4
DEFAULT_EMBEDDING_DIM = 384  # 2560 for Nucleotide Transformer output TODO

# Baseline analysis
DEFAULT_TEST_SIZE = 0.2
DEFAULT_COMPONENTS = 5
DEFAULT_MIN_COUNT = 2  # Minimum count for a label to be considered valid
DEFAULT_CV = 5  # Number of cross-validation folds

# Inference
DEFAULT_MODEL = "InstaDeepAI/nucleotide-transformer-2.5b-multi-species"
DEFAULT_CHUNK_SIZE = 4000
DEFAULT_STRIDE_RATIO = 0.5  # 50% overlap
DEFAULT_MAX_LEN = (
    45_000  # Max length for CONTIGS, to avoid a sparse matrix upon padding
)

# Inference and training
DEFAULT_NONCBL_LABEL = "NON-CBL"

# Sketching
DEFAULT_K = 15
DEFAULT_SKETCH_SIZE = 2**14

# k-NN classification
DEFAULT_TEST_SIZE = 0.2
DEFAULT_KNN_K = 5
DEFAULT_MIN_COUNT = 2  # Minimum count for a label to be considered valid

# Novelty detection
DEFAULT_MIN_SEROGROUP_SIZE = 40  # Minimum number of samples in a serogroup to be considered for novelty detection
DEFAULT_ENERGY_TEMPERATURE = 1.0  # Temperature T used in energy calculation

# Training contrastive transformer
DEFAULT_LR = 2e-5
DEFAULT_EPOCHS = 100
DEFAULT_EARLY_STOPPING = 10
DEFAULT_TEMPERATURE = 0.07
DEFAULT_WEIGHT_FINE = 1.0
DEFAULT_WEIGHT_COARSE = 0.5
DEFAULT_CONTRASTIVE_LOSS_RATIO = 0.5
DEFAULT_KFOLDS = 5
