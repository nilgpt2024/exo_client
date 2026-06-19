from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import time

from .partitioning_strategy import PartitioningStrategy, Partition
from .topology import Topology
from exo.inference.shard import Shard, ModelLoadState


@dataclass
class ModelLocation:
    node_id: str
    shard: Shard
    last_used: float


class ModelAwarePartitioningStrategy(PartitioningStrategy):
    def __init__(self):
        self.model_locations: Dict[str, List[ModelLocation]] = {}

    def update_model_location(self, node_id: str, load_state: ModelLoadState):
        model_id = load_state.model_id
        if model_id not in self.model_locations:
            self.model_locations[model_id] = []
        
        new_location = ModelLocation(
            node_id=node_id,
            shard=load_state.shard,
            last_used=time.time()
        )
        
        existing = next((loc for loc in self.model_locations[model_id] 
                         if loc.node_id == node_id and loc.shard == load_state.shard), None)
        if existing:
            existing.last_used = time.time()
        else:
            self.model_locations[model_id].append(new_location)
            print(f"[ModelAwarePartition] Added model location: {model_id} on {node_id}")

    def remove_model_location(self, node_id: str, model_id: Optional[str] = None):
        if model_id:
            if model_id in self.model_locations:
                self.model_locations[model_id] = [
                    loc for loc in self.model_locations[model_id] if loc.node_id != node_id
                ]
                if not self.model_locations[model_id]:
                    del self.model_locations[model_id]
        else:
            for mid in list(self.model_locations.keys()):
                self.model_locations[mid] = [
                    loc for loc in self.model_locations[mid] if loc.node_id != node_id
                ]
                if not self.model_locations[mid]:
                    del self.model_locations[mid]

    def can_cover_model(self, model_id: str, n_layers: int) -> Tuple[bool, List[ModelLocation]]:
        if model_id not in self.model_locations:
            return False, []
        
        locations = self.model_locations[model_id]
        
        covered_layers = set()
        used_locations = []
        
        for loc in locations:
            loc_shard = loc.shard
            if loc_shard.n_layers != n_layers:
                print(f"[ModelAwarePartition] Skipping location {loc.node_id} - n_layers mismatch")
                continue
            
            for layer in range(loc_shard.start_layer, loc_shard.end_layer + 1):
                covered_layers.add(layer)
            used_locations.append(loc)
        
        all_layers = set(range(n_layers))
        if covered_layers == all_layers:
            print(f"[ModelAwarePartition] Model {model_id} can be fully covered by existing shards")
            return True, used_locations
        
        print(f"[ModelAwarePartition] Model {model_id} coverage incomplete. "
              f"Covered: {sorted(covered_layers)}, Missing: {sorted(all_layers - covered_layers)}")
        return False, []

    def _create_partitions_from_locations(self, locations: List[ModelLocation], 
                                          n_layers: int) -> List[Partition]:
        partitions = []
        
        for loc in locations:
            start = loc.shard.start_layer / n_layers
            end = (loc.shard.end_layer + 1) / n_layers
            partitions.append(Partition(loc.node_id, start, end))
        
        partitions.sort(key=lambda p: p.start)
        return partitions

    def _smart_partition(self, topology: Topology, model_id: Optional[str] = None, 
                        n_layers: Optional[int] = None) -> List[Partition]:
        from .ring_memory_weighted_partitioning_strategy import RingMemoryWeightedPartitioningStrategy
        
        ring_strategy = RingMemoryWeightedPartitioningStrategy()
        return ring_strategy.partition(topology, model_id, n_layers)

    def partition(self, topology: Topology, model_id: Optional[str] = None, 
                 n_layers: Optional[int] = None) -> List[Partition]:
        if model_id and n_layers:
            print(f"[ModelAwarePartition] Checking coverage for {model_id} ({n_layers} layers)")
            
            can_cover, locations = self.can_cover_model(model_id, n_layers)
            if can_cover:
                print(f"[ModelAwarePartition] Reusing existing shards for {model_id}")
                for loc in locations:
                    loc.last_used = time.time()
                return self._create_partitions_from_locations(locations, n_layers)
        
        print(f"[ModelAwarePartition] Falling back to smart partitioning for {model_id}")
        return self._smart_partition(topology, model_id, n_layers)
