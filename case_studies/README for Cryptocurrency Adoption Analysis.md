# Cryptocurrency Adoption Analysis using IHP Framework

## Overview

This project implements a comprehensive analysis of global cryptocurrency adoption using the Integrative Historical Prediction (IHP) framework. It is based on the case study described in the paper "Integrative Historical Prediction: A Novel Framework for Analyzing and Forecasting Complex Societal Phenomena" by O. Austegard and C. Sonnett.

The analysis integrates multiple data sources, employs advanced mathematical modeling techniques, and leverages machine learning to forecast potential cryptocurrency adoption scenarios over the next 5-10 years.

## Features

- Data integration from diverse sources including blockchain data, economic indicators, social media sentiment, regulatory information, and financial institution activities
- Network analysis of the cryptocurrency ecosystem
- Agent-based modeling of adoption decisions
- Wavelet analysis for multi-scale temporal patterns
- Predictive modeling using Multiscale Temporal Convolutional Networks (MTCN)
- Regulatory landscape analysis
- Identification of potential tipping points in institutional adoption
- Volatility trend analysis

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/crypto-adoption-analysis.git
   cd crypto-adoption-analysis
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
   python crypto_adoption_analysis.py
   ```

3. The script will output a summary of the results, including:
   - Network properties of the cryptocurrency ecosystem
   - Predicted adoption rate by 2027
   - Regulatory analysis (crypto-friendly vs. crypto-strict countries)
   - Potential tipping point for institutional adoption
   - Predicted volatility by 2026

## Data Sources

This analysis requires the following data files:

- `blockchain_data.csv`: Transaction volumes and new wallet creation rates
- `economic_indicators.csv`: GDP, inflation rates, and currency stability indices
- `social_media_data.json`: Sentiment analysis from Twitter, Reddit, etc.
- `regulatory_info.csv`: Cryptocurrency regulations by country
- `financial_institutions.csv`: Crypto-related activities of major financial institutions

Ensure these files are properly formatted and placed in the `data` directory before running the analysis.

## Customization

You can customize the analysis by modifying the following parameters in the `crypto_adoption_analysis.py` file:

- `manifold_dim`: Dimension of the Riemannian manifold (default: 100)
- `num_variables`: Number of variables in the societal dynamics engine (default: 50)
- `num_agents`: Number of agents in the agent-based model (default: 1,000,000)
- `decision_threshold`: Threshold for adoption decision in the agent-based model (default: 0.7)

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

Project Link: https://github.com/yourusername/crypto-adoption-analysis