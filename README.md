# NevoRL

NevoRL is a Universal Vectorized Reinforcement Learning Domain-Specific Language (DSL) that compiles scalar agent/environment descriptions into high-throughput, auto-resetting Gymnasium-compatible VectorEnvs.

## Architecture Overview

The `nevoRL` project consists of a single Python file, `nevoRL.py`, which defines the DSL and its parsing logic. The core components are:

1. **Grammar Definition**: A Lark grammar is defined to parse environment descriptions.
2. **Parsing and Compilation**: The parsed descriptions are transformed into Gymnasium-compatible VectorEnvs.

## Tech Stack

- **Programming Language**: Python
- **Libraries**:
  - `numpy`: For numerical operations.
  - `lark`: For parsing the DSL grammar.

## Getting Started / Setup Guide

### Prerequisites

Ensure you have Python installed on your system. It is recommended to use a virtual environment for dependency management.

```bash
# Create and activate a virtual environment (optional but recommended)
python -m venv nevo-env
source nevo-env/bin/activate  # On Windows use `nevo-env\Scripts\activate`

# Install required libraries
pip install numpy lark-parser gymnasium
```

### Usage

1. **Define an Environment**: Create a `.nvrl` file with your environment description using the NevoRL DSL.

2. **Compile and Run**: Use `nevoRL.py` to compile and run your environment.

```bash
# Example command (adjust paths as necessary)
python nevoRL.py path/to/your/environment.nvrl
```

### Example Environment Description

Create a file named `example.nvrl` with the following content:

```nvrl
env MyEnv {
    state {
        position: float;
        velocity: float;
    }

    action discrete(3);

    observation {
        return [position, velocity];
    }

    reset {
        position = 0.0;
        velocity = 0.0;
    }

    step(action) {
        # Update logic here
    }

    reward {
        # Reward calculation here
    }
}
```

Run the environment using:

```bash
python nevoRL.py example.nvrl
```

This will compile and execute your custom environment defined in the DSL.