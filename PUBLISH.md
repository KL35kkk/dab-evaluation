# Publishing dab-evaluation to PyPI

## Prerequisites

1. **PyPI Account**: Create an account at [PyPI](https://pypi.org/account/register/)
2. **TestPyPI Account**: Create an account at [TestPyPI](https://test.pypi.org/account/register/)
3. **API Token**: Generate an API token from your PyPI account settings

## Setup

1. **Install build tools**:
   ```bash
   pip install --upgrade pip
   pip install build twine
   ```

2. **Configure credentials** (optional):
   ```bash
   # Create ~/.pypirc file
   [distutils]
   index-servers = pypi testpypi

   [pypi]
   username = __token__
   password = pypi-your-api-token-here

   [testpypi]
   repository = https://test.pypi.org/legacy/
   username = __token__
   password = pypi-your-testpypi-token-here
   ```

## Publishing Steps

### 1. Clean and Build

```bash
# Clean previous builds
rm -rf build/ dist/ *.egg-info/

# Build the package
python -m build
```

### 2. Check the Build

```bash
# Check for common issues
twine check dist/*
```

### 3. Upload to TestPyPI (Recommended First)

```bash
# Upload to test PyPI first
twine upload --repository testpypi dist/*

# Test install from test PyPI
pip install -i https://test.pypi.org/simple/ dab-evaluation
```

### 4. Upload to Production PyPI

```bash
# Upload to production PyPI
twine upload dist/*
```

## Verification

After publishing, verify the package:

```bash
# Install from PyPI
pip install dab-evaluation

# Test import
python -c "import dab_eval; print(dab_eval.__version__)"
```

## Package Structure

```
dab-evaluation/
├── dab_eval/
│   ├── __init__.py
│   ├── dab_eval.py
│   └── evaluation/
│       ├── __init__.py
│       ├── base_evaluator.py
│       ├── llm_evaluator.py
│       └── hybrid_evaluator.py
├── data/
│   └── benchmark.csv
├── examples/
│   ├── basic_usage.py
│   └── batch_evaluation.py
├── setup.py
├── pyproject.toml
├── requirements.txt
├── README.md
├── LICENSE
└── MANIFEST.in
```

## Usage After Installation

```python
import asyncio
from dab_eval import DABEvaluator, AgentMetadata, TaskCategory

async def main():
    # LLM configuration
    llm_config = {
        "model": "gpt-4",
        "base_url": "https://api.openai.com/v1",
        "api_key": "your-api-key",
        "temperature": 0.3,
        "max_tokens": 2000
    }
    
    # Create evaluator
    evaluator = DABEvaluator(llm_config, "output")
    
    # Define agent
    agent = AgentMetadata(
        url="http://localhost:8000",
        capabilities=[TaskCategory.WEB_RETRIEVAL],
        timeout=30
    )
    
    # Evaluate agent
    result = await evaluator.evaluate_agent(
        question="What is the date when Bitcoin was created?",
        agent_metadata=agent,
        category=TaskCategory.WEB_RETRIEVAL,
        expected_answer="2009-01-03"
    )
    
    print(f"Score: {result.evaluation_score}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Troubleshooting

### Common Issues

1. **"Package already exists"**: Increment version number in `setup.py` and `pyproject.toml`
2. **"Invalid distribution"**: Check `MANIFEST.in` and ensure all files are included
3. **"Missing dependencies"**: Update `requirements.txt` and `pyproject.toml`

### Version Management

- Update version in both `setup.py` and `pyproject.toml`
- Use semantic versioning (e.g., 1.0.0, 1.0.1, 1.1.0)
- Tag releases in git: `git tag v1.0.0 && git push origin v1.0.0`
