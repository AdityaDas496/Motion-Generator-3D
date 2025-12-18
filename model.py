import torch
import torch.nn as nn
import math
from transformers import CLIPModel

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        # We need to transpose x from [batch, seq, dim] to [seq, batch, dim]
        x = x.transpose(0, 1)
        x = x + self.pe[:x.size(0)]
        x = self.dropout(x)
        # Transpose back
        return x.transpose(0, 1)


# The architecture of the AI model
class MotionTransformer(nn.Module):
    def __init__(self, motion_features, d_model=512, nhead=8, num_encoder_layers=6, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        # Text Encoder (CLIP)
        print("Loading CLIP text model...")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        # Freezing the inner neural layers since we only need to train the outer layer
        for param in self.clip_model.parameters():
            param.requires_grad = False
            
        # Input/Output Layers (Projectors)
        # CLIP's text output is 512 features
        self.text_projector = nn.Linear(512, d_model)
        
        # 221 Mapping the 512 text output of the CLIP model to the 221 features of our data
        self.motion_projector = nn.Linear(motion_features, d_model)
        
        # Transformer Encoder
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # This is the standard Transformer "block"
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # This makes it accept [batch, seq, features]
        )
        # We stack 6 of these blocks to make the model a deep neural network model
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # The "Head" (Output Layer)
        # This takes the Transformer's 512-dim "thoughts" and projects
        # them back down to the 221-dim motion features
        self.output_head = nn.Linear(d_model, motion_features)

    def forward(self, text_inputs, motion_data):
        # text_inputs: The tokenized text from the collator
        # motion_data: The padded motion batch [batch, seq_len, 221]
        
        # This gives [batch_size, 512]
        text_features = self.clip_model.get_text_features(**text_inputs)
        
        # Project text features
        text_embed = self.text_projector(text_features)
        
        # Project motion features
        motion_embed = self.motion_projector(motion_data)
        
        # Add "page numbers" (Positional Encoding)
        motion_embed = self.pos_encoder(motion_embed)
        
        # Combine text and motion
        # We "condition" the motion by adding the text's "meaning"
        # to every single frame.
        # [batch_size, 512] -> [batch_size, 1, 512]
        text_embed = text_embed.unsqueeze(1)
        
        # [batch, seq, 512] + [batch, 1, 512] = [batch, seq, 512] (via broadcasting)
        combined_embed = motion_embed + text_embed

        # Run it through the "Brain"
        transformer_output = self.transformer_encoder(combined_embed)
        
        # Project back to motion
        final_output = self.output_head(transformer_output)
        
        return final_output
