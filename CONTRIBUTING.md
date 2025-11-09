# Contributing to Materials-Chemistry

Thank you for your interest in contributing to this educational resource! This guide will help you contribute effectively.

## How to Contribute

### Adding a New Notebook

1. **Fork the repository** and create a new branch for your contribution.

2. **Choose the appropriate topic directory** in `notebooks/` or create a new one if needed.

3. **Create your notebook** following these guidelines:
   - Use a descriptive filename (e.g., `ionic_bonding_visualization.ipynb`)
   - Start with a clear title and learning objectives
   - Include explanatory text with markdown cells
   - Add code cells with well-commented examples
   - Include visualizations where appropriate
   - End with exercises or practice problems
   - Add a summary or key takeaways section

4. **Test your notebook**:
   - Run all cells from top to bottom with a fresh kernel
   - Ensure all outputs are visible
   - Check that all dependencies are in `requirements.txt`
   - Verify all links and references work

5. **Submit a pull request** with:
   - A clear description of what the notebook teaches
   - The target audience (e.g., undergraduate, graduate)
   - Any new dependencies added

### Improving Existing Notebooks

- Fix typos, errors, or unclear explanations
- Add additional examples or visualizations
- Improve code efficiency or readability
- Update deprecated code or libraries

### Notebook Standards

#### Structure
- **Title**: Clear, descriptive title
- **Introduction**: Brief overview of the topic
- **Learning Objectives**: What students will learn
- **Prerequisites**: Required background knowledge
- **Content**: Step-by-step explanations with code
- **Exercises**: Practice problems (optional)
- **Summary**: Key takeaways
- **References**: Citations for theory or data

#### Code Style
- Follow PEP 8 style guidelines for Python code
- Use meaningful variable names
- Add comments for complex operations
- Keep code cells focused (one concept per cell when possible)

#### Markdown Style
- Use headers to organize content (##, ###, etc.)
- Include equations using LaTeX syntax when needed
- Add images and diagrams to clarify concepts
- Use bullet points and numbered lists for clarity

### Adding Data Files

- Place datasets in the `data/` directory
- Use open-source or public domain data when possible
- Include a `README.md` in `data/` describing each dataset
- Keep file sizes reasonable (< 10 MB if possible)
- Document data sources and licenses

### Adding Images

- Place images in the `images/` directory
- Use descriptive filenames
- Prefer vector formats (SVG) when possible
- Optimize image sizes for web viewing
- Include attribution if the image is not original

## Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Focus on educational value
- Credit sources and collaborators

## Questions?

If you have questions about contributing, please open an issue on GitHub and tag it with "question".

## Review Process

1. Submissions are reviewed by maintainers
2. Feedback may be provided for improvements
3. Once approved, your contribution will be merged
4. You'll be credited as a contributor

Thank you for helping make chemistry education more accessible and engaging!
