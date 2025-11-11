# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TrigoRL is a reinforcement learning laboratory for training AI agents to play Trigo, a 3D variant of Go. The project consists of:

1. **Trigo Game Engine** (third_party/trigo submodule) - Complete TypeScript implementation with Vue 3 frontend
2. **RL Training Framework** (to be implemented) - PyTorch + Transformers + wandb + ONNX

The game engine is fully functional. The RL framework is planned but not yet implemented.

## Common Commands

### Game Engine Development (third_party/trigo/trigo-web)

**Installation:**
```bash
cd third_party/trigo/trigo-web
npm run install:all              # Install all dependencies
npm run build:parsers            # Build TGN parser (required before first run)
```

**Development:**
```bash
npm run dev                      # Start frontend + backend
npm run dev:app                  # Frontend only (http://localhost:5173)
npm run dev:backend              # Backend only (http://localhost:3000)
```

**Building:**
```bash
npm run build                    # Build both
npm run build:app                # Build frontend
npm run build:backend            # Build backend
```

**Testing:**
```bash
npm test                         # Watch mode
npm run test:run                 # Run once
npm exec vitest -- run tests/game/trigoGame.core.test.ts  # Single test file
```

**Tools:**
```bash
npm run generate:games           # Generate random TGN game files
npm run generate:games -- --count 100 --moves 20-80
npm run format                   # Format with Prettier
```

**Parser Generation:**
The TGN parser is built from Jison grammar (`inc/tgn/tgn.jison`). Rebuild after grammar changes:
```bash
npm run build:parser:tgn
```

## Architecture

### Component Hierarchy

```
TrigoRL (Root)
└── third_party/trigo/trigo-web/
    ├── inc/                     # Shared TypeScript (game logic)
    │   ├── trigo/               # Core game engine
    │   └── tgn/                 # TGN parser
    ├── app/                     # Vue 3 frontend (Three.js rendering)
    ├── backend/                 # Express + Socket.io server
    └── tests/                   # Vitest test suite (109 tests)
```

### Game Engine Flow

**Game State Management:**
```
TrigoGame Class (inc/trigo/game.ts)
├── Board State: 3D array Stone[x][y][z]
├── Move History: Array of Step objects
├── Current Step Index: Position in history (enables undo/redo/jump)
├── Ko Detection: Last captured positions
└── Territory Cache: Calculated territories
```

**Frontend Architecture:**
```
User Input → gameStore (Pinia) → TrigoGameFrontend → TrigoGame
                ↓                                        ↓
         Vue Reactivity                         Core Game Logic
                ↓                                        ↓
         trigoViewport ← Board State ← gameUtils (capture, Ko)
                ↓
         Three.js → WebGL
```

**Multiplayer Synchronization:**
```
Client 1                Backend                    Client 2
   │                       │                          │
   │─ makeMove ──────────→ │                          │
   │                    GameManager                   │
   │                   (TrigoGame.drop)               │
   │                       │                          │
   │← gameUpdate ──────────┼────────→ gameUpdate ────┤
   │   (Socket.io)         │          (Socket.io)    │
```

**TGN Parsing:**
```
TGN String → Jison Parser → AST → TrigoGame.fromTGN() → Replay Moves
```

### Key Technical Concepts

**TrigoGame Class** (`inc/trigo/game.ts`):
- Central state manager for all game logic
- Maintains 3D board array and complete move history
- Implements all Go rules: capture detection, Ko rule, suicide prevention, territory calculation
- Supports undo/redo and jump to any step in history
- Serializes to/from JSON and TGN format

**TGN Coordinate System** (`inc/trigo/ab0yz.ts`):
- Center-symmetric notation where `0` = center on each axis
- From edges: `a, b, c, ...` (one edge), `z, y, x, ...` (opposite edge)
- Examples (5×5×5 board):
  - `000` = center (2,2,2)
  - `aaa` = corner (0,0,0)
  - `zzz` = opposite corner (4,4,4)
- 2D boards omit trailing 1s (19×19×1 uses 2-char coords like `aa`)
- Functions: `encodeAb0yz()` and `decodeAb0yz()`

**Capture Detection in 3D** (`inc/trigo/gameUtils.ts`):
1. Each position has up to 6 neighbors (±x, ±y, ±z)
2. Groups formed by connected stones of same color (flood fill)
3. Liberties are empty spaces adjacent to a group
4. Groups with 0 liberties after move are captured
5. Algorithm:
   - Place stone on temp board
   - Find neighboring enemy groups
   - Check each for 0 liberties
   - Remove captured groups
   - Store captured positions in history

**Ko Rule Implementation**:
- Prevents immediately recapturing a single stone if it recreates the previous position
- Check conditions:
  1. Previous move captured exactly 1 stone
  2. Current move would capture exactly 1 stone
  3. Placing at the previously captured position
- If all true: Ko violation (move rejected)
- Storage: `lastCapturedPositions` in TrigoGame

### Critical Files

**Core Game Logic:**
- `inc/trigo/game.ts` - TrigoGame class (main engine)
- `inc/trigo/gameUtils.ts` - Go rules, capture, Ko, territory
- `inc/trigo/types.ts` - TypeScript interfaces
- `inc/trigo/ab0yz.ts` - TGN coordinate encoding

**State Management:**
- `app/src/stores/gameStore.ts` - Pinia store
- `app/src/utils/TrigoGameFrontend.ts` - Frontend wrapper

**3D Rendering:**
- `app/src/services/trigoViewport.ts` - Three.js viewport manager
- `app/src/views/TrigoView.vue` - Main game view

**Parser:**
- `inc/tgn/tgn.jison` - Jison grammar for TGN
- `inc/trigo/parserInit.ts` - Parser initialization

**Backend:**
- `backend/src/server.ts` - Express + Socket.io server
- `backend/src/services/gameManager.ts` - Multi-room manager
- `backend/src/sockets/gameSocket.ts` - WebSocket handlers

## Development Conventions

**From Trigo Submodule:**
- Learn development history from `agentlog.md` first
- Update `agentlog.md` when mini-milestones are accomplished
- Use **camelCase** for .ts file naming
- Use multiple blank lines to separate logical code sections (2 lines between top-level blocks)
- Place new .md documents in `./docs/`, always in English
- Check `CLAUDE.local.md` for additional instructions

**TypeScript Patterns:**
- Strict type safety throughout
- Interface-based design
- Enum for stone types: 0=Empty, 1=Black, 2=White
- Position objects: `{x, y, z}` for coordinates
- Type adapters for frontend/backend conversion

**Testing:**
- Vitest with `describe`/`it`/`expect` pattern
- `beforeEach()` for game setup
- Descriptive test names
- Group related tests in `describe()` blocks
- Current status: 109/109 tests passing (100%)

**Code Style (Prettier):**
- Tab indentation
- Double quotes for strings
- Semicolons always

## RL Framework Implementation Plan

The following components need to be built for the RL training framework:

**1. Environment Wrapper:**
- Python interface to Trigo game engine
- OpenAI Gym-compatible environment
- State representation for 3D board
- Action space definition (valid moves)
- Integration approaches:
  - JSON API via backend server
  - Direct TGN file I/O
  - ONNX for model inference in game engine

**2. Model Architecture:**
- Transformer-based policy network
- Value estimation network
- Feature extraction from 3D board state
- Attention mechanism for spatial relationships

**3. Training Pipeline:**
- Self-play game generation
- Experience replay buffer
- PPO or actor-critic algorithm
- Weights & Biases (wandb) integration
- Hyperparameter tuning

**4. Model Export:**
- ONNX format for cross-platform deployment
- Inference optimization
- Integration with game engine

**5. Evaluation:**
- Elo rating system
- Game quality metrics
- Visualization tools

## Integration Points

**Game Engine → RL Framework:**
- TGN format for game serialization/loading
- JSON API for Python-TypeScript communication
- Backend GameManager can orchestrate self-play
- Board state extraction from TrigoGame class

**RL Framework → Game Engine:**
- ONNX models deployed for AI player inference
- REST/WebSocket API for move generation
- Training data saved as TGN files

## Testing

**Test Suite Location:** `third_party/trigo/trigo-web/tests/game/`

**Test Categories:**
- `trigoGame.core.test.ts` - Basic operations (35 tests)
- `trigoGame.history.test.ts` - Undo/redo/jump (21 tests)
- `trigoGame.rules.test.ts` - Go rules, Ko, suicide (18 tests)
- `trigoGame.state.test.ts` - State persistence (32 tests)
- `trigoGame.tgn.test.ts` - TGN import/export
- `trigoGame.fromTGN.test.ts` - TGN parsing
- `verify_capture.test.ts` - Capture verification
- `debug_*.test.ts` - Debug scenarios

## Resources

- [Trigo Game Engine](https://github.com/k-l-lambda/trigo)
- [TGN Format Specification](third_party/trigo/docs/tgn-format-spec.md)
- [Game Engine README](third_party/trigo/README.md)


## Meta-Instructions

**Important constraints to remember**:
1. Learn the development history from `agentlog.md` firstly.
1. Update `agentlog.md` when a mini-milestone is accomplished.
1. Use camel case for .py file naming.
1. Use multiple space lines in code to separate sections according to logic levels. Use 2 lines for interval between top-level code blocks.
1. Place new .md documents in `./docs/`, documents in English always.
1. Following further instuctions in `CLAUDE.local.md` if present.
