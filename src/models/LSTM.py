import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

@dataclass
class LSTMConfig:
    """Configuration for LSTM model"""
    input_size: int
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    bidirectional: bool = True
    attention_heads: int = 4
    use_layer_norm: bool = True
    residual_connections: bool = True
    confidence_threshold: float = 0.6  # Threshold for high-confidence predictions
    
    def validate(self):
        """Validate configuration parameters"""
        assert self.input_size > 0, "Input size must be positive"
        assert self.hidden_size > 0, "Hidden size must be positive"
        assert self.num_layers > 0, "Number of layers must be positive"
        assert 0 <= self.dropout < 1, "Dropout must be between 0 and 1"
        assert self.attention_heads > 0, "Number of attention heads must be positive"
        assert 0.5 <= self.confidence_threshold < 1, "Confidence threshold must be between 0.5 and 1"

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism"""
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert hidden_size % num_heads == 0, "Hidden size must be divisible by number of heads"
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        
        # Linear layers for query, key, value projections
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        
        # Output projection
        self.output = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_length, _ = x.size()
        
        # Apply layer normalization first (pre-norm formulation)
        x_norm = self.layer_norm(x)
        
        # Linear projections and reshape for multi-head attention
        q = self.query(x_norm).view(batch_size, seq_length, self.num_heads, self.head_size)
        k = self.key(x_norm).view(batch_size, seq_length, self.num_heads, self.head_size)
        v = self.value(x_norm).view(batch_size, seq_length, self.num_heads, self.head_size)
        
        # Transpose for attention computation
        q = q.transpose(1, 2)  # (batch_size, num_heads, seq_length, head_size)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_size, dtype=torch.float32))
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Attention weights
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        # Compute context vectors
        context = torch.matmul(attention, v)
        
        # Reshape and apply output projection
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_length, self.hidden_size)
        output = self.output(context)
        
        # Residual connection
        return x + self.dropout(output)

class LSTMLayer(nn.Module):
    """Single LSTM layer with layer normalization and residual connection"""
    def __init__(self, input_size: int, hidden_size: int, dropout: float = 0.1, 
                 bidirectional: bool = True, use_layer_norm: bool = True):
        super().__init__()
        self.use_layer_norm = use_layer_norm
        self.hidden_size = hidden_size
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size // 2 if bidirectional else hidden_size,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Layer normalization
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(hidden_size)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, 
                hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # Apply LSTM
        output, (h_n, c_n) = self.lstm(x, hx)
        
        # Apply layer normalization if enabled
        if self.use_layer_norm:
            output = self.layer_norm(output)
        
        # Apply dropout
        output = self.dropout(output)
        
        return output, (h_n, c_n)

class DirectionalLSTM(nn.Module):
    """
    Improved LSTM model with:
    - Bidirectional LSTM layers
    - Multi-head self-attention
    - Layer normalization
    - Residual connections
    - Gradient clipping
    """
    def __init__(self, config: LSTMConfig):
        super().__init__()
        config.validate()
        self.config = config
        
        print(f"[DEBUG] Initializing DirectionalLSTM with config input_size={config.input_size}")
        
        # We'll initialize the input_norm in the forward pass
        self.input_norm = None
        self.input_size_set = False
        self.real_input_size = None  # Will store actual input size from first forward pass
        
        # Hold off on initializing LSTM layers until we know the real input size
        self.lstm_layers = None
        self.attention = None
        self.final_norm = None
        self.dense = None
        self.output = None

    def _initialize_layers(self, input_size: int):
        """Initialize layers with the correct input size and dtype"""
        print(f"[DEBUG] Initializing layers with actual input_size={input_size}")
        
        # Set default tensor type to float32
        torch.set_default_tensor_type(torch.FloatTensor)
        
        # LSTM layers with residual connections
        self.lstm_layers = nn.ModuleList()
        current_input_size = input_size
        
        for i in range(self.config.num_layers):
            lstm_layer = LSTMLayer(
                input_size=current_input_size,
                hidden_size=self.config.hidden_size,
                dropout=self.config.dropout,
                bidirectional=self.config.bidirectional,
                use_layer_norm=self.config.use_layer_norm
            )
            self.lstm_layers.append(lstm_layer)
            current_input_size = self.config.hidden_size
        
        # Multi-head attention layer
        self.attention = MultiHeadAttention(
            hidden_size=self.config.hidden_size,
            num_heads=self.config.attention_heads,
            dropout=self.config.dropout
        )
        
        # Final prediction layers
        self.final_norm = nn.LayerNorm(self.config.hidden_size)
        self.dropout = nn.Dropout(self.config.dropout)
        self.dense = nn.Linear(self.config.hidden_size, self.config.hidden_size // 2)
        self.output = nn.Linear(self.config.hidden_size // 2, 2)
        
        # Ensure all parameters are float32
        self.type(torch.float32)
        
        # Move to same device if model is already on GPU
        if next(self.parameters(), None) is not None and next(self.parameters()).is_cuda:
            self.to(next(self.parameters()).device)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: [batch_size, sequence_length, feature_dim]
        batch_size, seq_len, features = x.size()
        
        # Initialize layers if this is the first forward pass
        if self.lstm_layers is None:
            self._initialize_layers(features)
            self.real_input_size = features
        
        # Verify input size hasn't changed
        if features != self.real_input_size:
            raise ValueError(f"Input feature dimension changed. Expected {self.real_input_size}, got {features}")
        
        # Initialize input_norm with the correct feature dimension
        if not hasattr(self, 'input_size_set') or not self.input_size_set:
            self.input_norm = nn.BatchNorm1d(features)
            if next(self.parameters()).is_cuda:
                self.input_norm = self.input_norm.cuda()
            self.input_size_set = True
        
        # Apply input normalization
        # Reshape to [batch_size * seq_len, features] for BatchNorm1d
        x = x.reshape(-1, features)
        x = self.input_norm(x)
        # Reshape back to [batch_size, seq_len, features]
        x = x.reshape(batch_size, seq_len, features)
        
        # Process LSTM layers with residual connections
        for i, lstm_layer in enumerate(self.lstm_layers):
            if i > 0 and self.config.residual_connections:
                residual = x
            
            # Apply LSTM layer - input: [batch_size, seq_len, features]
            x, _ = lstm_layer(x)  # We're not using the hidden states, so we can ignore them
            
            # Add residual connection if enabled
            if i > 0 and self.config.residual_connections:
                x = x + residual
        
        # Apply attention mechanism
        # Input: [batch_size, seq_len, hidden_size]
        x = self.attention(x)
        
        # Final processing
        x = self.final_norm(x)
        x = self.dropout(x)
        
        # Get last sequence output - select last timestep
        # From: [batch_size, seq_len, hidden_size] to [batch_size, hidden_size]
        x = x[:, -1, :]
        
        # Dense layers
        x = F.relu(self.dense(x))
        x = self.dropout(x)
        
        # Output layer
        x = self.output(x)
        # Apply softmax for probabilities
        probabilities = F.softmax(x, dim=-1)
        
        return probabilities

    def predict_with_confidence(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Make predictions with confidence scores
        
        Returns:
            Tuple containing:
            - predicted direction (0 for down, 1 for up)
            - confidence score (probability of predicted direction)
            - raw probabilities for both directions
        """
        probabilities = self(x)
        
        # Get predicted direction and confidence
        predicted_direction = torch.argmax(probabilities, dim=1)
        confidence = torch.max(probabilities, dim=1).values
        
        return predicted_direction, confidence, probabilities

    def configure_optimizers(self, learning_rate: float = 1e-3, 
                           weight_decay: float = 1e-5) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
        """Configure optimizer and learning rate scheduler"""
        # Optimizer with weight decay
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=learning_rate,
            epochs=1000,
            steps_per_epoch=100,
            pct_start=0.3,
            anneal_strategy='cos'
        )
        
        return optimizer, scheduler
    
    def train_model(self, train_loader, validation_loader, epochs=100, learning_rate=1e-3, 
             weight_decay=1e-5, early_stopping_params=None, class_weights=None,
             device='cuda' if torch.cuda.is_available() else 'cpu') -> Dict[str, float]:
        """
        Custom training method with validation and early stopping
        """
        self.to(device)
    
        # Initialize layers by doing a forward pass with the first batch
        print("[DEBUG] Initializing model layers...")
        try:
            first_batch = next(iter(train_loader))
            first_data = first_batch[0].to(device).float()  # Explicitly convert to float32
            _ = self(first_data)
            print("[DEBUG] Model layers initialized successfully")
        except Exception as e:
            print(f"[DEBUG] Error during layer initialization: {str(e)}")
            raise
        
        self.train()
        
        # Now that layers are initialized, configure optimizer and scheduler
        print("[DEBUG] Configuring optimizer and scheduler...")
        try:
            optimizer, scheduler = self.configure_optimizers(learning_rate, weight_decay)
            print("[DEBUG] Optimizer and scheduler configured successfully")
        except Exception as e:
            print(f"[DEBUG] Error configuring optimizer: {str(e)}")
            raise
        
        # Setup loss function with class weights if provided
        if class_weights is not None:
            weight_tensor = torch.tensor([class_weights[i] for i in range(len(class_weights))],
                                    dtype=torch.float32,  # Explicitly use float32
                                    device=device)
            criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        else:
            criterion = nn.CrossEntropyLoss()
        
        # Initialize early stopping
        best_val_loss = float('inf')
        patience = early_stopping_params.get('patience', 10) if early_stopping_params else 10
        patience_counter = 0
        best_model_state = None
        
        # Training metrics
        metrics = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': []
        }
        
        print("[DEBUG] Starting training loop...")
        for epoch in range(epochs):
            # Training phase
            self.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data = data.to(device).float()  # Explicitly convert to float32
                target = target.to(device)
                
                optimizer.zero_grad()
                output = self(data)
                loss = criterion(output, target)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                train_total += target.size(0)
                train_correct += (predicted == target).sum().item()
            
            avg_train_loss = train_loss / len(train_loader)
            train_accuracy = 100. * train_correct / train_total
            
            # Validation phase
            self.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for data, target in validation_loader:
                    data, target = data.to(device), target.to(device)
                    output = self(data)
                    loss = criterion(output, target)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(output.data, 1)
                    val_total += target.size(0)
                    val_correct += (predicted == target).sum().item()
            
            avg_val_loss = val_loss / len(validation_loader)
            val_accuracy = 100. * val_correct / val_total
            
            # Update metrics
            metrics['train_loss'].append(avg_train_loss)
            metrics['val_loss'].append(avg_val_loss)
            metrics['train_accuracy'].append(train_accuracy)
            metrics['val_accuracy'].append(val_accuracy)
            
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%')
            print(f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')
            
            # Early stopping check
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = self.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'Early stopping triggered after {epoch+1} epochs')
                    break
        
        # Restore best model
        if best_model_state is not None:
            self.load_state_dict(best_model_state)
        
        # Add final metrics
        metrics['best_val_loss'] = best_val_loss
        metrics['final_train_loss'] = avg_train_loss
        metrics['final_train_accuracy'] = train_accuracy
        metrics['final_val_accuracy'] = val_accuracy
        metrics['epochs_trained'] = epoch + 1
        
        return metrics