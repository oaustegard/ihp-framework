# Arab Spring Analysis using IHP Framework

## Overview

This project implements a comprehensive analysis of the Arab Spring using the Integrative Historical Prediction (IHP) framework. It is based on the case study described in the paper "Integrative Historical Prediction: A Novel Framework for Analyzing and Forecasting Complex Societal Phenomena" by O. Austegard and C. Sonnett.

The analysis integrates multiple data sources, employs advanced mathematical modeling techniques, and leverages machine learning to provide insights into the complex dynamics of the Arab Spring.

## Features

- Data integration from diverse sources including social media, economic indicators, political events, media coverage, and demographic data
- Event pattern analysis using persistent homology
- Multi-scale analysis using Multiscale Temporal Convolutional Networks (MTCN)
- Digital idea transmission analysis through social network modeling
- Economic-social media-protest nexus analysis
- Precursor signal detection using Riemannian geometry
- Media attention dynamics analysis

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/arab-spring-analysis.git
   cd arab-spring-analysis
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Ensure all required data files are in the `data` directory.

2. Run the main analysis script:
   ```
   python arab_spring_analysis.py
   ```

3. The script will output a summary of the results, including:
   - Number of influential events identified
   - Number of communities detected
   - Betti numbers from topological analysis
   - Prediction accuracy
   - Hashtags analyzed
   - Economic-Social Media-Protest Nexus Correlation
   - Number of detected precursor signals
   - Countries analyzed for media attention dynamics

## Data Sources

This analysis requires the following data files:

- `social_media_data.csv`: Twitter and Facebook data
- `economic_indicators.csv`: GDP, unemployment, and inflation data
- `political_events.json`: Data on political events during the Arab Spring
- `media_coverage.csv`: Data on media coverage of the events
- `demographic_data.csv`: Population statistics and age distributions

Ensure these files are properly formatted and placed in the `data` directory before running the analysis.

## Customization

You can customize the analysis by modifying the following parameters in the `arab_spring_analysis.py` file:

- `manifold_dim`: Dimension of the Riemannian manifold (default: 100)
- `num_variables`: Number of variables in the societal dynamics engine (default: 50)
- `num_layers`, `kernel_size`, `num_filters`, `dropout_rate`: Parameters for the MTCN

## Contributing

Contributions to improve the analysis or extend its capabilities are welcome. Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/amazing-feature`)
3. Make your changes and commit them (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Citation

If you use this implementation in your research, please cite the original paper:

```
Austegard, O., & Sonnett, C. (2024). Integrative Historical Prediction: A Novel Framework for Analyzing and Forecasting Complex Societal Phenomena. Journal of Complex Systems Analysis, 15(2), 123-456.
```

## Contact

For questions or feedback, please contact:

Your Name - your.email@example.com

Project Link: https://github.com/yourusername/arab-spring-analysis