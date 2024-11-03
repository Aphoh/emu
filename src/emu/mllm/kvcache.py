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
        """Returns the sequence length of the cached states for the given layer."""
        if len(self.key_cache) > layer_idx and isinstance(self.key_cache[layer_idx], torch.Tensor):
            return self.key_cache[layer_idx].shape[-2]
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
            cache_kwargs: Dictionary containing cache_position tensor indicating where to insert states
            
        Returns:
            Tuple of the updated key and value states
        """
        cache_position = None if cache_kwargs is None else cache_kwargs.get('cache_position')
        if cache_position is None:
            raise ValueError("cache_position is required in cache_kwargs")
            
        # Update seen tokens count for first layer
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]
            
        # Initialize layer caches if needed
        while len(self.key_cache) <= layer_idx:
            self.key_cache.append([])
            self.value_cache.append([])
            
        if len(self.key_cache[layer_idx]) == 0:
            # First insertion for this layer - need to pad to chunk size
            max_position = cache_position.max().item() + 1
            needed_size = ((max_position - 1) // self.chunk_size + 1) * self.chunk_size
            
            # Create padded tensors
            padded_shape = list(key_states.shape)
            padded_shape[-2] = needed_size
            
            new_key_cache = torch.zeros(padded_shape, dtype=key_states.dtype, device=key_states.device)
            new_value_cache = torch.zeros(padded_shape, dtype=value_states.dtype, device=value_states.device)
            
            # Insert initial states at specified positions
            new_key_cache[..., cache_position, :] = key_states
            new_value_cache[..., cache_position, :] = value_states
            
            self.key_cache[layer_idx] = new_key_cache
            self.value_cache[layer_idx] = new_value_cache
            
            return new_key_cache[..., :max_position, :], new_value_cache[..., :max_position, :]
        else:
            current_length = self.key_cache[layer_idx].shape[-2]
            max_position = cache_position.max().item() + 1
            
            # If we need to insert beyond current size or are at last position, expand
            if max_position >= current_length:
                needed_size = ((max_position - 1) // self.chunk_size + 1) * self.chunk_size
                
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
            
            # Insert new states at specified positions
            self.key_cache[layer_idx][..., cache_position, :] = key_states
            self.value_cache[layer_idx][..., cache_position, :] = value_states
            
            # Return up to the highest used position
            return_length = max(current_length, max_position)
            return (self.key_cache[layer_idx][..., :return_length, :],
                   self.value_cache[layer_idx][..., :return_length, :])