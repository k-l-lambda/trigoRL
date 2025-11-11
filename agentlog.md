
> This is a reinforcement learning lab project. Based on the game of Trigo, a board game of Go in 3D space.
> Technical basic: pyTorch + transformers, wanDB for training stats, onnx exporting for model weights support.


## 2025/11/11


> Plan the framework for this project, investigate the configuration system in `../deep-starry` project.
> Set up modernized deep learning training basic modules based on a similar YAML config system to deep-starry.

<details>
<summary>Complete RL training framework implemented</summary>

Investigated deep-starry's configuration system and implemented a modernized version for TrigoRL:

**Configuration System**:
- Hydra + OmegaConf for composable YAML configs (modernization from deep-starry's pure PyYAML)
- Hierarchical config structure: env + agent + training
- CLI overrides supported
- Auto-generated experiment directories

**Core Infrastructure**:
- Agent registry with factory pattern (inspired by deep-starry)
- Environment registry for flexible env creation
- WandbLogger for experiment tracking (modernization from TensorBoard)
- CheckpointManager with best model tracking

**Training System**:
- Custom RLTrainer adapted from deep-starry's Trainer
- Episode-based training loop (vs epoch-based)
- Evaluation episodes with metric monitoring
- Checkpoint save/load with state persistence

**Initial Components**:
- RandomAgent for testing pipeline
- DummyEnv for framework validation
- MLP and PolicyValueNetwork architectures

**Project Structure**:
```
trigoRL/
├── configs/              # Hydra configuration files
├── trigor/             # Main package
│   ├── agents/          # Agent registry and implementations
│   ├── envs/            # Environment registry and wrappers
│   ├── training/        # Custom RL trainer
│   ├── models/          # Neural network architectures
│   └── utils/           # Logger, checkpoint manager
├── train.py             # Main entry point
└── requirements.txt     # Dependencies
```

Next steps: Test the framework end-to-end with random agent + dummy environment.
</details>
