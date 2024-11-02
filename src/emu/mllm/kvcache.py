import torch
from typing import List, Tuple, Dict, Any, Optional

class ChunkedDynamicCache:
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
        
    def current_size(self) -> int:
        """Returns the current allocated size of the cache in tokens."""
        if not self.key_cache or not isinstance(self.key_cache[0], torch.Tensor):
            return 0
        return self.key_cache[0].shape[-2]
        
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
            # First insertion for this layer
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            current_length = self.key_cache[layer_idx].shape[-2]
            new_length = current_length + key_states.shape[-2]
            
            # Calculate needed chunks
            needed_size = ((new_length - 1) // self.chunk_size + 1) * self.chunk_size
            
            if needed_size > current_length:
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
            
            # Add new states
            self.key_cache[layer_idx][..., current_length:new_length, :] = key_states
            self.value_cache[layer_idx][..., current_length:new_length, :] = value_states
            
        return self.key_cache[layer_idx], self.value_cache[layer_idx]