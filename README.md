<div align="center">

# CodeFlow: Predicting Program Behavior with Dynamic Dependencies Learning
[![arXiv](https://img.shields.io/badge/arXiv-2408.02816-b31b1b.svg)](https://arxiv.org/abs/2408.02816)

</div>

## Introduction

We introduce **CodeFlow**, a novel machine learning-based approach designed to predict program behavior by learning both static and dynamic dependencies within the code. CodeFlow constructs control flow graphs (CFGs) to represent all possible execution paths and uses these graphs to predict code coverage and detect runtime errors. Our empirical evaluation demonstrates that CodeFlow significantly improves code coverage prediction accuracy and effectively localizes runtime errors, outperforming state-of-the-art models.

### Paper: [CodeFlow: Predicting Program Behavior with Dynamic Dependencies Learning](https://arxiv.org/abs/2408.02816)

## Installation

To set up the environment and install the necessary libraries, run the following command:

```sh
./setup.sh
```

## Architecture
<img src="img/architecture.png">

CodeFlow consists of several key components:
1. **CFG Building**: Constructs CFGs from the source code.
2. **Source Code Representation Learning**: Learns vector representations of CFG nodes.
3. **Dynamic Dependencies Learning**: Captures dynamic dependencies among statements using execution traces.
4. **Code Coverage Prediction**: Classifies nodes for code coverage using learned embeddings.
5. **Runtime Error Detection and Localization**: Detects and localizes runtime errors by analyzing code coverage continuity within CFGs.

## Usage

### Running CodeFlow Model

To run the CodeFlow model, use the following command:

```sh
python main.py --data <dataset> [--runtime_detection] [--bug_localization]
```

#### Configuration Options

- `--data`: Specify the dataset to be used for training. Options:
  - `CodeNet`: Train with only non-buggy Python code from the CodeNet dataset.
  - `FixEval_complete`: Train with both non-buggy and buggy code from the FixEval and CodeNet dataset.
  - `FixEval_incomplete`: Train with the incomplete version of the FixEval_complete dataset.

- `--runtime_detection`: Validate the Runtime Error Detection.

- `--bug_localization`: Validate the Bug Localization in buggy code.

#### Example Usage

1. **Training with the CodeNet dataset(RQ1):**
    ```sh
    python main.py --data CodeNet
    ```

2. **Training with the complete FixEval dataset and validating Runtime Error Detection(RQ2):**
    ```sh
    python main.py --data FixEval_complete --runtime_detection
    ```

3. **Training with the complete and incomplete FixEval dataset and validating Bug Localization(RQ3):**
    ```sh
    python main.py --data FixEval_complete --bug_localization
    python main.py --data FixEval_incomplete --bug_localization
    ```
### Fuzz Testing with LLM Integration (RQ4)

After training CodeFlow and saving the corresponding checkpoint, you can utilize it for fuzz testing by integrating it with a Large Language Model (LLM). Use the following command:

```sh
python fuzz_testing.py --checkpoint <number> --epoch <number> --time <seconds> --claude_api_key <api_key> --model <model_name>
```
- `checkpoint`: The chosen checkpoint.
- `epoch`: The chosen epoch of checkpoint.
- `time`: Time in seconds to run fuzz testing for each code file.
- `claude_api_key`: Your API key for Claude.
- `model`: Model of Claude, default is claude-3-5-sonnet-20240620.
#### Example
```sh
python fuzz_testing.py --checkpoint 1 --epoch 600 --time 120 --claude_api_key YOUR_API_KEY --model claude-3-5-sonnet-20240620
```
### Generating Your Own Dataset

To generate your own dataset, including CFG, forward and backward edges, and the true execution trace as ground truth for your Python code, follow these steps:

1. **Navigate to the `generate_dataset` folder**:
    ```sh
    cd generate_dataset
    ```

2. **Place your Python code files in the `dataset` folder**.

3. **Run the dataset generation script**:
    ```sh
    python generate_dataset.py
    ```
To build and visualize CFG for a Python file, use this command:
```sh
python cfg.py \directory_to_Python_file
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements
This codebase is adapted from:
- [ConditionBugs](https://github.com/zhangj111/ConditionBugs)
- [CFG-Generator](https://github.com/Tiankai-Jiang/CFG-Generator)
- [trace_python](https://github.com/python/cpython/blob/3.12/Lib/trace.py)

## Citation Information

If you're using CodeFlow, please cite using this BibTeX:
```bibtex
@misc{le2024learningpredictprogramexecution,
      title={Learning to Predict Program Execution by Modeling Dynamic Dependency on Code Graphs}, 
      author={Cuong Chi Le and Hoang Nhat Phan and Huy Nhat Phan and Tien N. Nguyen and Nghi D. Q. Bui},
      year={2024},
      eprint={2408.02816},
      archivePrefix={arXiv},
      primaryClass={cs.SE},
      url={https://arxiv.org/abs/2408.02816}, 
}
```
