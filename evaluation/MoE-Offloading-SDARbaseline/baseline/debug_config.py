# ============================================================================
# AdaPEagle Unified Debug Configuration
# ============================================================================
# Centralized debug flags for Qwen3-MoE and GPT-OSS
# Set flags to True/False to enable/disable specific debug output

# ============================================================================
# Model Building Debug
# ============================================================================
PRINT_MODEL_BUILD = False         # Print model building progress
PRINT_BUFFER_INIT = False         # Print expert buffer initialization
PRINT_WEIGHT_LOADING = False      # Print weight loading progress

# ============================================================================
# MoE Feature Enable/Disable
# ============================================================================
MOE_DEBUG_ENABLED = False         # Enable MoE activation data collection

# ============================================================================
# Ablation Study Controls
# ============================================================================
PREFETCH_ENABLED = True           # Enable router input collection and expert prefetching
BMM_ENABLED = True                # BMM batch optimization (vs. per-expert loop)
PRELAUNCH_ENABLED = True          # Async compute launch (vs. sync wait for load)

# ============================================================================
# Expert Activation Debug
# ============================================================================
PRINT_EXPERT_DETAILS = False      # Print detailed expert activation info (causes small memcpyDtoH)

# ============================================================================
# Prefetch Debug
# ============================================================================
PRINT_PREFETCH_DEBUG = False      # Print prefetch-related debug info

# ============================================================================
# Buffer Management Debug
# ============================================================================
PRINT_BUFFER_DEBUG = False        # Print expert buffer management debug

# ============================================================================
# Token Tree Debug (EAGLE3)
# ============================================================================
DEBUG_TOKEN_TREE = False          # Print DFT model token tree analysis (cnets.py)
DEBUG_TOKEN_TREE_MOE = False      # Print token tree + MoE expert information (ea_model.py, tests/)

# ============================================================================
# Verification Debug
# ============================================================================
DEBUG_VERIFICATION = False        # Print verification results and expert utilization

# ============================================================================
# Data Collection
# ============================================================================
COLLECT_TOKEN_TREE_DATA = False   # Collect per-token expert data for token tree analysis

# ============================================================================
# Test Output Control
# ============================================================================
PRINT_GENERATION_RESULT = False    # Print generation result text

# ============================================================================
# GPU Cache Configuration
# ============================================================================
PRINT_CACHE_UPDATE_DEBUG = False   # Print cache update statistics per forward
