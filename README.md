# TrigoRL

A reinforcement learning laboratory project for training AI agents to play Trigo, a 3D variant of the board game Go.

## Overview

TrigoRL is an experimental platform for exploring reinforcement learning techniques in the context of **Trigo**
- a strategic board game that extends the rules of Go into three-dimensional space.
While traditional Go is played on a 2D 19×19 board, Trigo is played on a cubic grid,
introducing new strategic dimensions and complexity.

## About Trigo

Trigo is a modern reimplementation of a 3D Go variant with the following characteristics:

- **Board**: 3D cubic grid (default: 5×5×5, configurable to other dimensions including 2D boards)
- **Rules**: Based on Go mechanics adapted for 3D space
  - Stone placement with capture detection
  - Ko rule enforcement
  - Territory calculation in 3D
  - Pass, undo/redo, and resignation support
- **Notation**: TGN (Trigo Game Notation) - a PGN-inspired text format for recording games
- **Coordinate System**: Center-symmetric notation (e.g., `000` = center, `aaa` = corner)

**TRY IT YOURSELF ONLINE**: here is a [Trigo demo page](https://huggingface.co/spaces/k-l-lambda/trigo).

## Technical Stack

### Reinforcement Learning Framework (Planned)

- **PyTorch**: Deep learning framework for model implementation
- **Transformers**: Architecture foundation for the RL agent
- **Weights & Biases (wandb)**: Training metrics and experiment tracking
- **ONNX**: Model weight export format for cross-platform deployment

## Development Roadmap

The following components need to be implemented for the RL framework:

1. **Environment Wrapper**
   - Python interface to the Trigo game engine
   - OpenAI Gym-compatible environment
   - State representation for 3D board positions
   - Action space definition

2. **Model Architecture**
   - Transformer-based policy network
   - Value estimation network
   - Feature extraction from 3D board state

3. **Training Pipeline**
   - Self-play game generation
   - Experience replay buffer
   - Policy gradient or actor-critic implementation
   - Integration with Weights & Biases for experiment tracking

4. **Model Export**
   - ONNX conversion utilities
   - Inference optimization

5. **Evaluation & Analysis**
   - Agent performance metrics
   - Game quality assessment
   - Visualization tools

## Game Engine Features

The Trigo game engine provides:

- **3D Visualization**: Interactive Three.js-based board rendering
- **Multiplayer Support**: Real-time gameplay via WebSocket
- **Game Notation**: TGN format for saving and loading games
- **REST API**: Programmatic game control
- **Comprehensive Testing**: 10 test suites covering core functionality

For detailed API documentation, see:
- [Game Engine README](third_party/trigo/README.md)
- [TGN Format Specification](third_party/trigo/docs/tgn-format-spec.md)
- [Development Guidelines](third_party/trigo/CLAUDE.md)

## Acknowledgments

- Based on the Trigo game engine by k-l-lambda
- Inspired by AlphaGo and other game-playing RL systems
