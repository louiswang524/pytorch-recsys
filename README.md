# PyTorch Recommendation System Framework

A modular, high-performance PyTorch-based framework for sequential recommendation systems with support for transformer-based models and optimized attention mechanisms.

## Overview

This framework provides a complete solution for building and training sequential recommendation models, specifically designed to handle long sequences (up to 1000 items) efficiently. It features optimized attention mechanisms, comprehensive data loading infrastructure, and production-ready serving utilities.

## Key Features

- **🚀 Optimized Attention**: Linear and sparse attention mechanisms for long sequences
- **🏗️ Modular Architecture**: Clean separation of data, models, training, and evaluation
- **⚡ High Performance**: Optimized for sequences up to 1000 items with efficient memory usage  
- **🔧 Easy Configuration**: Hydra-based configuration system for reproducible experiments
- **📊 Comprehensive Evaluation**: Standard recommendation metrics (NDCG, Recall, MRR)
- **🎯 Production Ready**: Model optimization and serving utilities included

## Supported Models

- **SASRec**: Self-Attentive Sequential Recommendation with optimized attention
- **TransAct**: Advanced transformer architecture with linear attention mechanisms

## Quick Start

```bash
# Clone the repository
git clone https://github.com/louiswang524/pytorch-recsys.git
cd pytorch-recsys

# Install dependencies
pip install -e .

# Run example training
python examples/train_sasrec.py --config-name=sasrec_movielens
```

## Project Status

This framework is currently under active development. Track progress on our [GitHub Issues](https://github.com/louiswang524/pytorch-recsys/issues/1).

**Development Timeline**: 2 weeks
**Current Phase**: Core Infrastructure (Phase 1/4)

### Development Phases

1. **Phase 1: Core Infrastructure** - Project structure, base models, data loading, configuration
2. **Phase 2: Model Implementation** - SASRec and TransAct with optimized attention
3. **Phase 3: Training & Evaluation** - PyTorch Lightning integration, metrics framework
4. **Phase 4: Optimization & Polish** - Model optimization, serving utilities, documentation

## Architecture

```
pytorch-recsys/
├── pytorch_recsys/
│   ├── data/          # Data loading and preprocessing
│   ├── models/        # Model implementations (SASRec, TransAct)
│   ├── layers/        # Reusable components (attention, embeddings)
│   ├── training/      # PyTorch Lightning training infrastructure
│   ├── evaluation/    # Metrics and evaluation framework
│   ├── serving/       # Model serving utilities
│   └── configs/       # Hydra configuration files
├── examples/          # Usage examples and tutorials
├── tests/            # Comprehensive test suite
└── docs/             # Documentation
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- PyTorch Lightning 2.0+
- Hydra Core 1.3+

## Contributing

This project follows a spec-driven development approach with full traceability from requirements to code. See our [development workflow](https://github.com/louiswang524/pytorch-recsys/issues/1) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{pytorch_recsys_framework,
  title={PyTorch Recommendation System Framework},
  author={Your Name},
  year={2025},
  url={https://github.com/louiswang524/pytorch-recsys}
}
```