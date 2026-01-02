"""
Common Dataset Utilities Module.

This module provides common utilities and base classes for
dataset handling across different benchmarks.
"""

from typing import Any, Dict, Iterator, List, Optional
from dataclasses import dataclass, field


@dataclass
class DatasetItem:
    """Base class for dataset items."""
    id: str
    input_text: str
    reference: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseDataset:
    """
    Base class for dataset wrappers.
    """
    
    def __init__(self, name: str):
        """
        Initialize base dataset.
        
        Args:
            name: Dataset name.
        """
        self.name = name
        self.data: List[DatasetItem] = []
        self._loaded = False
    
    def load(self) -> None:
        """Load dataset. Override in subclasses."""
        raise NotImplementedError
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __iter__(self) -> Iterator[DatasetItem]:
        return iter(self.data)
    
    def __getitem__(self, idx: int) -> DatasetItem:
        return self.data[idx]
    
    @property
    def is_loaded(self) -> bool:
        return self._loaded


def preprocess_dataset(
    data: List[Dict],
    text_field: str = "text",
    label_field: Optional[str] = None
) -> List[DatasetItem]:
    """
    Preprocess raw dataset into DatasetItem format.
    
    Args:
        data: Raw data list.
        text_field: Field name for input text.
        label_field: Optional field name for labels.
        
    Returns:
        List of DatasetItem objects.
    """
    # TODO: Implement preprocessing
    items = []
    for i, row in enumerate(data):
        item = DatasetItem(
            id=str(i),
            input_text=row.get(text_field, ""),
            reference=row.get(label_field) if label_field else None,
            metadata=row,
        )
        items.append(item)
    return items


def split_dataset(
    data: List[DatasetItem],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42
) -> Dict[str, List[DatasetItem]]:
    """
    Split dataset into train/val/test.
    
    Args:
        data: Dataset items.
        train_ratio: Training set ratio.
        val_ratio: Validation set ratio.
        seed: Random seed.
        
    Returns:
        Dictionary with 'train', 'val', 'test' splits.
    """
    # TODO: Implement dataset splitting
    import random
    random.seed(seed)
    
    shuffled = data.copy()
    random.shuffle(shuffled)
    
    n = len(shuffled)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    return {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }


def sample_dataset(
    data: List[DatasetItem],
    n: int,
    seed: int = 42
) -> List[DatasetItem]:
    """
    Sample n items from dataset.
    
    Args:
        data: Dataset items.
        n: Number of samples.
        seed: Random seed.
        
    Returns:
        Sampled items.
    """
    # TODO: Implement sampling
    import random
    random.seed(seed)
    return random.sample(data, min(n, len(data)))
