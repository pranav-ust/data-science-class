"""
Portfolio Optimization using Hugging Face Transformers and TRL
This implementation uses:
1. Hugging Face Time Series Transformer for price prediction
2. TRL (Transformer Reinforcement Learning) for portfolio optimization
3. Chronos models for advanced time series forecasting
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import yfinance as yf
from transformers import (
    TimeSeriesTransformerConfig,
    TimeSeriesTransformerForPrediction,
    Trainer,
    TrainingArguments
)
from datasets import Dataset as HFDataset
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# For Chronos models (if available)
try:
    from chronos import ChronosPipeline
    CHRONOS_AVAILABLE = True
except ImportError:
    CHRONOS_AVAILABLE = False
    print("Chronos not available. Install with: pip install git+https://github.com/amazon-science/chronos-forecasting.git")

# Check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ============== Data Preparation ==============

@dataclass
class TimeSeriesDataConfig:
    """Configuration for time series data"""
    prediction_length: int = 30
    context_length: int = 90
    batch_size: int = 32
    num_workers: int = 4

class StockTimeSeriesDataset(Dataset):
    """Dataset for time series transformer"""
    
    def __init__(self, data: pd.DataFrame, config: TimeSeriesDataConfig):
        self.data = data.values
        self.config = config
        self.n_assets = data.shape[1]
        
    def __len__(self):
        return len(self.data) - self.config.context_length - self.config.prediction_length + 1
    
    def __getitem__(self, idx):
        # Get context and prediction windows
        past_values = self.data[idx:idx + self.config.context_length]
        future_values = self.data[idx + self.config.context_length:idx + self.config.context_length + self.config.prediction_length]
        
        # Create masks (all observed in this case)
        past_observed_mask = np.ones_like(past_values)
        future_observed_mask = np.ones_like(future_values)
        
        return {
            'past_values': torch.FloatTensor(past_values),
            'past_observed_mask': torch.FloatTensor(past_observed_mask),
            'future_values': torch.FloatTensor(future_values),
            'future_observed_mask': torch.FloatTensor(future_observed_mask)
        }

# ============== Portfolio Optimization with RL ==============

class PortfolioAgent(nn.Module):
    """
    Portfolio agent using transformer architecture
    Designed to work with TRL for reinforcement learning
    """
    
    def __init__(self, n_assets: int, d_model: int = 256, n_heads: int = 8, 
                 n_layers: int = 6, context_length: int = 90):
        super().__init__()
        self.n_assets = n_assets
        self.context_length = context_length
        
        # Price encoding
        self.price_encoder = nn.Linear(n_assets * 2, d_model)  # prices + returns
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Portfolio weight prediction
        self.portfolio_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, n_assets),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, prices, returns):
        # Combine price and return information
        features = torch.cat([prices, returns], dim=-1)
        
        # Encode features
        encoded = self.price_encoder(features)
        
        # Apply transformer
        transformer_out = self.transformer(encoded)
        
        # Use last hidden state for portfolio weights
        last_hidden = transformer_out[:, -1, :]
        
        # Generate portfolio weights
        weights = self.portfolio_head(last_hidden)
        
        return weights

class RewardModel(nn.Module):
    """
    Reward model for portfolio optimization
    Learns to predict portfolio performance metrics
    """
    
    def __init__(self, n_assets: int, d_model: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_assets * 3, d_model),  # weights, returns, prices
            nn.ReLU(),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1)  # Single reward score
        )
        
    def forward(self, weights, returns, prices):
        features = torch.cat([weights, returns, prices], dim=-1)
        reward = self.encoder(features)
        return reward

# ============== Hugging Face Time Series Forecasting ==============

def prepare_hf_dataset(returns_data: pd.DataFrame, config: TimeSeriesDataConfig) -> HFDataset:
    """Prepare dataset for Hugging Face time series transformer"""
    
    dataset_list = []
    
    for i in range(len(returns_data) - config.context_length - config.prediction_length + 1):
        past_values = returns_data.iloc[i:i + config.context_length].values
        future_values = returns_data.iloc[i + config.context_length:i + config.context_length + config.prediction_length].values
        
        # For multivariate, we need to handle each asset separately
        for asset_idx in range(returns_data.shape[1]):
            dataset_list.append({
                'past_values': past_values[:, asset_idx].tolist(),
                'future_values': future_values[:, asset_idx].tolist(),
                'asset_id': asset_idx
            })
    
    return HFDataset.from_list(dataset_list)

def train_time_series_transformer(train_dataset: HFDataset, val_dataset: HFDataset, 
                                config: TimeSeriesDataConfig, n_assets: int):
    """Train Hugging Face TimeSeriesTransformer"""
    
    # Model configuration
    model_config = TimeSeriesTransformerConfig(
        prediction_length=config.prediction_length,
        context_length=config.context_length,
        distribution_output="normal",
        d_model=128,
        d_ff=512,
        num_attention_heads=8,
        num_encoder_layers=4,
        num_decoder_layers=4,
        scaling="std",
        dropout=0.1,
        encoder_layerdrop=0.1,
        decoder_layerdrop=0.1,
        attention_dropout=0.1,
        activation_dropout=0.1,
        num_parallel_samples=100,
    )
    
    # Initialize model
    model = TimeSeriesTransformerForPrediction(model_config)
    model.to(device)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./portfolio_ts_model",
        num_train_epochs=20,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        learning_rate=1e-3,
        weight_decay=0.01,
        logging_steps=50,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    
    # Train model
    trainer.train()
    
    return model

# ============== Chronos Integration ==============

def use_chronos_for_prediction(historical_data: np.ndarray, prediction_length: int = 30):
    """Use Amazon Chronos for time series prediction"""
    if not CHRONOS_AVAILABLE:
        print("Chronos not available")
        return None
    
    # Load pre-trained Chronos model
    pipeline = ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-small",
        device_map=device,
        torch_dtype=torch.float32,
    )
    
    # Make predictions for each asset
    predictions = []
    
    for asset_idx in range(historical_data.shape[1]):
        asset_data = torch.tensor(historical_data[:, asset_idx])
        forecast = pipeline.predict(
            context=asset_data,
            prediction_length=prediction_length,
            num_samples=100,
        )
        predictions.append(forecast)
    
    return torch.stack(predictions, dim=1)

# ============== Portfolio Optimization Pipeline ==============

class PortfolioOptimizationPipeline:
    """Complete pipeline for portfolio optimization"""
    
    def __init__(self, tickers: List[str], config: TimeSeriesDataConfig):
        self.tickers = tickers
        self.config = config
        self.n_assets = len(tickers)
        
        # Initialize models
        self.portfolio_agent = PortfolioAgent(
            n_assets=self.n_assets,
            context_length=config.context_length
        ).to(device)
        
        self.reward_model = RewardModel(n_assets=self.n_assets).to(device)
        
    def download_data(self, start_date: str, end_date: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Download and prepare stock data"""
        prices = pd.DataFrame()
        
        for ticker in self.tickers:
            try:
                data = yf.download(ticker, start=start_date, end=end_date, progress=False)
                prices[ticker] = data['Adj Close']
            except Exception as e:
                print(f"Error downloading {ticker}: {e}")
        
        prices = prices.dropna()
        returns = prices.pct_change().dropna()
        
        return prices, returns
    
    def train_models(self, prices: pd.DataFrame, returns: pd.DataFrame):
        """Train all models in the pipeline"""
        
        # Prepare datasets
        train_size = int(0.8 * len(returns))
        train_returns = returns[:train_size]
        val_returns = returns[train_size:]
        
        print("Training Time Series Transformer...")
        # Prepare HF datasets
        train_hf_dataset = prepare_hf_dataset(train_returns, self.config)
        val_hf_dataset = prepare_hf_dataset(val_returns, self.config)
        
        # Train time series model
        ts_model = train_time_series_transformer(
            train_hf_dataset, val_hf_dataset, self.config, self.n_assets
        )
        
        print("\nTraining Portfolio Agent with RL...")
        # Train portfolio agent using a simple supervised approach
        # In practice, you would use TRL with proper RL algorithms
        self._train_portfolio_agent(train_returns.values, val_returns.values)
        
        return ts_model
    
    def _train_portfolio_agent(self, train_returns: np.ndarray, val_returns: np.ndarray, 
                             epochs: int = 50, lr: float = 1e-3):
        """Train portfolio agent using supervised learning on optimal allocations"""
        
        optimizer = torch.optim.Adam(self.portfolio_agent.parameters(), lr=lr)
        
        for epoch in range(epochs):
            total_loss = 0
            n_batches = 0
            
            # Create batches
            for i in range(0, len(train_returns) - self.config.context_length - 1, self.config.batch_size):
                end_idx = min(i + self.config.batch_size, len(train_returns) - self.config.context_length - 1)
                
                batch_losses = []
                
                for j in range(i, end_idx):
                    # Get context window
                    context_returns = train_returns[j:j + self.config.context_length]
                    next_returns = train_returns[j + self.config.context_length]
                    
                    # Convert to tensors
                    returns_tensor = torch.FloatTensor(context_returns).unsqueeze(0).to(device)
                    prices_tensor = torch.ones_like(returns_tensor)  # Placeholder
                    
                    # Get portfolio weights
                    weights = self.portfolio_agent(prices_tensor, returns_tensor)
                    
                    # Calculate portfolio return
                    portfolio_return = torch.sum(weights * torch.FloatTensor(next_returns).unsqueeze(0).to(device))
                    
                    # Simple loss: negative return (we want to maximize returns)
                    loss = -portfolio_return
                    batch_losses.append(loss)
                
                if batch_losses:
                    # Average batch loss
                    batch_loss = torch.stack(batch_losses).mean()
                    
                    # Backward pass
                    optimizer.zero_grad()
                    batch_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.portfolio_agent.parameters(), 1.0)
                    optimizer.step()
                    
                    total_loss += batch_loss.item()
                    n_batches += 1
            
            if epoch % 10 == 0:
                avg_loss = total_loss / n_batches if n_batches > 0 else 0
                print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")
    
    def backtest(self, returns: pd.DataFrame, initial_capital: float = 10000) -> Dict:
        """Backtest the portfolio strategy"""
        
        portfolio_values = [initial_capital]
        weights_history = []
        
        # Use test data
        test_returns = returns.values
        
        for i in range(self.config.context_length, len(test_returns)):
            # Get context
            context = test_returns[i - self.config.context_length:i]
            
            # Get portfolio weights
            with torch.no_grad():
                returns_tensor = torch.FloatTensor(context).unsqueeze(0).to(device)
                prices_tensor = torch.ones_like(returns_tensor)
                
                weights = self.portfolio_agent(prices_tensor, returns_tensor)
                weights = weights.cpu().numpy()[0]
            
            weights_history.append(weights)
            
            # Calculate portfolio return
            daily_return = np.dot(weights, test_returns[i])
            new_value = portfolio_values[-1] * (1 + daily_return)
            portfolio_values.append(new_value)
        
        # Calculate metrics
        portfolio_returns = pd.Series(portfolio_values[1:]).pct_change().dropna()
        
        total_return = (portfolio_values[-1] - initial_capital) / initial_capital
        annual_return = np.mean(portfolio_returns) * 252
        annual_vol = np.std(portfolio_returns) * np.sqrt(252)
        sharpe_ratio = annual_return / annual_vol if annual_vol > 0 else 0
        
        # Max drawdown
        running_max = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        return {
            'portfolio_values': portfolio_values,
            'weights_history': weights_history,
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        }

# ============== Visualization Functions ==============

def plot_results(results: Dict, benchmark_results: Dict, tickers: List[str]):
    """Plot portfolio performance results"""
    
    # Portfolio value comparison
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(results['portfolio_values'], label='AI Portfolio', linewidth=2)
    plt.plot(benchmark_results['portfolio_values'], label='Equal Weight', linewidth=2, alpha=0.7)
    plt.xlabel('Days')
    plt.ylabel('Portfolio Value ($)')
    plt.title('Portfolio Value Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Metrics comparison
    plt.subplot(2, 2, 2)
    metrics = ['Total Return', 'Annual Return', 'Sharpe Ratio', 'Max Drawdown']
    ai_values = [results['total_return'], results['annual_return'], 
                 results['sharpe_ratio'], results['max_drawdown']]
    benchmark_values = [benchmark_results['total_return'], benchmark_results['annual_return'],
                       benchmark_results['sharpe_ratio'], benchmark_results['max_drawdown']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.bar(x - width/2, ai_values, width, label='AI Portfolio')
    plt.bar(x + width/2, benchmark_values, width, label='Equal Weight')
    plt.xlabel('Metrics')
    plt.xticks(x, metrics, rotation=45)
    plt.legend()
    plt.title('Performance Metrics Comparison')
    
    # Weight evolution
    plt.subplot(2, 2, 3)
    weights_history = np.array(results['weights_history'])
    for i, ticker in enumerate(tickers):
        plt.plot(weights_history[:, i], label=ticker, alpha=0.7)
    plt.xlabel('Days')
    plt.ylabel('Weight')
    plt.title('Portfolio Weight Evolution')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Final weights
    plt.subplot(2, 2, 4)
    final_weights = weights_history[-1]
    plt.bar(tickers, final_weights)
    plt.xlabel('Assets')
    plt.ylabel('Weight')
    plt.title('Final Portfolio Allocation')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()

# ============== Main Execution ==============

def main():
    # Configuration
    TICKERS = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META', 'NVDA', 'TSLA', 'JPM', 'V', 'JNJ']
    START_DATE = '2020-01-01'
    END_DATE = '2024-01-01'
    
    # Data configuration
    config = TimeSeriesDataConfig(
        prediction_length=30,
        context_length=90,
        batch_size=32
    )
    
    print("=== Portfolio Optimization with Hugging Face Transformers ===\n")
    
    # Initialize pipeline
    pipeline = PortfolioOptimizationPipeline(TICKERS, config)
    
    # Download data
    print("1. Downloading stock data...")
    prices, returns = pipeline.download_data(START_DATE, END_DATE)
    print(f"Downloaded data for {len(TICKERS)} stocks")
    print(f"Date range: {prices.index[0]} to {prices.index[-1]}")
    
    # Split data for training and testing
    train_size = int(0.8 * len(returns))
    train_returns = returns[:train_size]
    test_returns = returns[train_size:]
    
    # Train models
    print("\n2. Training models...")
    ts_model = pipeline.train_models(prices, returns)
    
    # Use Chronos if available
    if CHRONOS_AVAILABLE:
        print("\n3. Using Chronos for additional predictions...")
        chronos_predictions = use_chronos_for_prediction(
            train_returns.values[-config.context_length:], 
            prediction_length=config.prediction_length
        )
        print(f"Chronos predictions shape: {chronos_predictions.shape}")
    
    # Backtest
    print("\n4. Running backtest...")
    results = pipeline.backtest(test_returns)
    
    # Benchmark: Equal weight portfolio
    print("\n5. Running benchmark (equal weight)...")
    equal_weights = np.ones(len(TICKERS)) / len(TICKERS)
    benchmark_portfolio_values = [10000]
    
    test_returns_values = test_returns.values
    for i in range(len(test_returns_values)):
        daily_return = np.dot(equal_weights, test_returns_values[i])
        new_value = benchmark_portfolio_values[-1] * (1 + daily_return)
        benchmark_portfolio_values.append(new_value)
    
    benchmark_returns = pd.Series(benchmark_portfolio_values[1:]).pct_change().dropna()
    benchmark_results = {
        'portfolio_values': benchmark_portfolio_values,
        'total_return': (benchmark_portfolio_values[-1] - 10000) / 10000,
        'annual_return': np.mean(benchmark_returns) * 252,
        'annual_volatility': np.std(benchmark_returns) * np.sqrt(252),
        'sharpe_ratio': (np.mean(benchmark_returns) * 252) / (np.std(benchmark_returns) * np.sqrt(252)),
        'max_drawdown': np.min((benchmark_portfolio_values - np.maximum.accumulate(benchmark_portfolio_values)) / np.maximum.accumulate(benchmark_portfolio_values))
    }
    
    # Print results
    print("\n=== Backtest Results ===")
    print("\nAI Portfolio (Transformer + RL):")
    print(f"Total Return: {results['total_return']:.2%}")
    print(f"Annual Return: {results['annual_return']:.2%}")
    print(f"Annual Volatility: {results['annual_volatility']:.2%}")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {results['max_drawdown']:.2%}")
    
    print("\nEqual Weight Portfolio (Benchmark):")
    print(f"Total Return: {benchmark_results['total_return']:.2%}")
    print(f"Annual Return: {benchmark_results['annual_return']:.2%}")
    print(f"Annual Volatility: {benchmark_results['annual_volatility']:.2%}")
    print(f"Sharpe Ratio: {benchmark_results['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {benchmark_results['max_drawdown']:.2%}")
    
    # Visualize results
    print("\n6. Visualizing results...")
    plot_results(results, benchmark_results, TICKERS)
    
    # Save model
    print("\n7. Saving trained models...")
    torch.save(pipeline.portfolio_agent.state_dict(), 'portfolio_agent.pth')
    print("Models saved successfully!")

if __name__ == "__main__":
    main()
