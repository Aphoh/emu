import torch
from typing import List, Tuple, Dict, Any, Optional
from transformers.cache_utils import Cache

class ChunkedDynamicCache(Cache):
    """
    A cache that grows dynamically in chunks of 128 tokens as more tokens are generated.
    The cache stores Key and Value states as lists of tensors, one for each layer.
    Each tensor has shape [batch_size, num_heads, seq_len, head_dim].
    """
    
    def __init__(self) -> None:
        self.chunk_size = 128
        self._seen_tokens = 0
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        
    def __len__(self):
        """Returns the number of layers in the cache."""
        return len(self.key_cache)
        
    def get_seq_length(self, layer_idx = 0):
        if len(self.key_cache) > layer_idx:
            return self._seen_tokens if layer_idx == 0 else self.key_cache[layer_idx].shape[-2]
        else:
            return 0
        
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with new key and value states, expanding in chunks of 128 tokens as needed.
        
        Args:
            key_states: New key states to cache
            value_states: New value states to cache
            layer_idx: Index of the layer to cache the states for
            cache_kwargs: Additional arguments (unused)
            
        Returns:
            Tuple of the updated key and value states
        """
        # Update seen tokens count for first layer
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]
            
        # Initialize layer caches if needed
        while len(self.key_cache) <= layer_idx:
            self.key_cache.append([])
            self.value_cache.append([])
            
        if len(self.key_cache[layer_idx]) == 0:
            # First insertion for this layer - need to pad to chunk size
            initial_length = key_states.shape[-2]
            needed_size = ((initial_length - 1) // self.chunk_size + 1) * self.chunk_size
            
            # Create padded tensors
            padded_shape = list(key_states.shape)
            padded_shape[-2] = needed_size
            
            new_key_cache = torch.zeros(padded_shape, dtype=key_states.dtype, device=key_states.device)
            new_value_cache = torch.zeros(padded_shape, dtype=value_states.dtype, device=value_states.device)
            
            # Copy initial states
            new_key_cache[..., :initial_length, :] = key_states
            new_value_cache[..., :initial_length, :] = value_states
            
            self.key_cache[layer_idx] = new_key_cache
            self.value_cache[layer_idx] = new_value_cache
            
            print(f"Initialized cache to size {needed_size}")
            return key_states, value_states
        else:
            current_length = self.key_cache[layer_idx].shape[-2]
            new_length = current_length + key_states.shape[-2]
            return_length = new_length  # Length of tensor to return
            
            # Calculate needed chunks - expand if we'll be at the last slot
            if new_length >= current_length:
                needed_size = ((new_length - 1) // self.chunk_size + 1) * self.chunk_size
                
                # Expand the cache to the next chunk boundary
                device = self.key_cache[layer_idx].device
                dtype = self.key_cache[layer_idx].dtype
                
                # Create expanded tensors
                expanded_shape = list(self.key_cache[layer_idx].shape)
                expanded_shape[-2] = needed_size
                
                new_key_cache = torch.zeros(expanded_shape, dtype=dtype, device=device)
                new_value_cache = torch.zeros(expanded_shape, dtype=dtype, device=device)
                
                # Copy existing cache
                new_key_cache[..., :current_length, :] = self.key_cache[layer_idx]
                new_value_cache[..., :current_length, :] = self.value_cache[layer_idx]
                
                self.key_cache[layer_idx] = new_key_cache
                self.value_cache[layer_idx] = new_value_cache
                print(f"Expanded cache to size {needed_size}")
            
            # Add new states
            self.key_cache[layer_idx][..., current_length:new_length, :] = key_states
            self.value_cache[layer_idx][..., current_length:new_length, :] = value_states
            
            # Return only the actually used portion
            return (self.key_cache[layer_idx][..., :return_length, :], 
                   self.value_cache[layer_idx][..., :return_length, :])