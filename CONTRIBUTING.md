# Contributing to Evaluator

First off, thank you for considering contributing to this project. It's people like you that make this project such a
great tool.

## Where do I go from here?

If you've noticed a bug or have a feature request, make sure to open a GitHub issue if one does not already exist. It's
always best to start with an issue, which can then be assigned to a developer to be worked on.

## Adding a New Model

If you want to add a new model to the project, please follow these steps:

1. **Fork & create a branch**: Fork the repository and create a branch with a descriptive name. A good branch name would be (where `new_model` is the name of the model you're adding):

    ```bash
    git checkout -b add-new_model
    ```

2. **Implement the Model**: Implement the new model in a separate Python file. Make sure to follow the existing code style and structure. Your model should have a `generate` method that takes a model name and a prompt as input and returns a generated response.

3. **Update the Evaluator**: Update the `evaluate` function in `src/evaluator.py` to handle the new model.

4. **Test Your Changes**: Make sure to thoroughly test your changes. This includes not only the new model itself but also the integration of the model with the rest of the project.

5. **Commit Your Changes**: Commit your changes and include a description of your changes in the commit.

6. **Submit a Pull Request**: Push your branch to your fork on GitHub, then press the New pull request button on GitHub. Please provide as much information as possible in the pull request description. Describe the changes you made, why you made them, how to test them, and anything else you think is important.

## Wait for Review

Once your pull request is submitted, a project maintainer will review your changes. Changes may be requested, and your
pull request will be approved once all requested changes have been made.

## Celebrate

Congratulations, you have just contributed to the HotpotQA Model Evaluator with Dria's Public RAG Model project!
