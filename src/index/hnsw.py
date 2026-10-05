import heapq
import math
import random
import numpy as np


class HNSW:
    def __init__(
        self,
        dim: int,
        M: int = 16,
        ef_construction: int = 200,
        seed: int = 42,
    ):
        """dim: embedding dimensionality (384 for your corpus)
        M: max bidirectional connections per node, per layer (layer 0 typically gets Mmax0 = 2*M)
        ef_construction: candidate list size during insertion — quality/build-time tradeoff
        """
        self.dim = dim
        self.M = M
        self.Mmax = M
        self.Mmax0 = M * 2  # paper's convention: layer 0 gets double the connections
        self.ef_construction = ef_construction
        self.ml = 1.0 / math.log(M)  # level-generation normalization factor, per the paper

        self.vectors: dict[int, np.ndarray] = {}  # node_id -> embedding
        self.layers: list[dict[int, set[int]]] = []  # layers[l][node_id] -> set of neighbor node_ids at layer l
        self.entry_point: int | None = None
        self.max_level: int = -1

        random.seed(seed)

    def _distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine distance for L2-normalized vectors: 1.0 - dot_product.
        
        Using distance (0.0 = identical, 2.0 = opposite) ensures that smaller is closer,
        aligning naturally with standard min-heap priority queues.
        """
        return float(1.0 - np.dot(a, b))

    def _get_random_level(self) -> int:
        """Exponentially-decaying random level assignment per Paper Algorithm 1:
        
        l = floor(-ln(uniform(0, 1)) * m_L)
        """
        r = random.random()
        # Prevent log(0) in edge case
        while r == 0:
            r = random.random()
        return math.floor(-math.log(r) * self.ml)

    def _search_layer(
        self, query: np.ndarray, entry_points: list[int], ef: int, layer: int
    ) -> list[tuple[float, int]]:
        """Greedy best-first search within a single layer (Algorithm 2 in paper).

        Maintains:
          - visited: set of evaluated node IDs
          - C (candidates): min-heap ordered by distance (closest candidate popped first)
          - W (found elements): max-heap ordered by distance (furthest element popped first)

        Returns:
          List of (distance, node_id) tuples ordered closest to furthest, capped at ef elements.
        """
        visited = set(entry_points)

        # Candidate heap C: min-heap -> (dist, node_id)
        candidates: list[tuple[float, int]] = []
        # Result set W: max-heap -> (-dist, node_id)
        w_max_heap: list[tuple[float, int]] = []

        for ep in entry_points:
            dist = self._distance(query, self.vectors[ep])
            heapq.heappush(candidates, (dist, ep))
            heapq.heappush(w_max_heap, (-dist, ep))

        while candidates:
            c_dist, c_node = heapq.heappop(candidates)
            f_dist = -w_max_heap[0][0]  # Furthest element distance currently in W

            # If current candidate is further than furthest element in W, break early
            if c_dist > f_dist:
                break

            # Explore neighbors of candidate c at specified layer
            neighbors = self.layers[layer].get(c_node, set())
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)

                    f_dist = -w_max_heap[0][0]
                    d_neighbor = self._distance(query, self.vectors[neighbor])

                    if d_neighbor < f_dist or len(w_max_heap) < ef:
                        heapq.heappush(candidates, (d_neighbor, neighbor))
                        heapq.heappush(w_max_heap, (-d_neighbor, neighbor))

                        if len(w_max_heap) > ef:
                            heapq.heappop(w_max_heap)  # Drop worst candidate

        # Convert max-heap to sorted list (closest first)
        results = sorted([(-neg_d, node) for neg_d, node in w_max_heap], key=lambda x: x[0])
        return results

    def _select_neighbors(
        self, candidates: list[tuple[float, int]], max_m: int
    ) -> list[int]:
        """Simple Heuristic (Algorithm 3): Selects the max_m closest candidates by distance."""
        # Ensure candidate list is sorted by ascending distance
        sorted_candidates = sorted(candidates, key=lambda x: x[0])
        return [node for _, node in sorted_candidates[:max_m]]

    def insert(self, node_id: int, vector: np.ndarray):
        """Full insert procedure (Algorithm 1 in paper)."""
        self.vectors[node_id] = vector
        new_level = self._get_random_level()

        # Step 1: Empty graph initialization
        if self.entry_point is None:
            self.entry_point = node_id
            self.max_level = new_level
            for l in range(new_level + 1):
                self.layers.append({node_id: set()})
            return

        # Ensure layers list has enough capacity
        while len(self.layers) <= max(self.max_level, new_level):
            self.layers.append({})

        curr_ep = [self.entry_point]
        curr_level = self.max_level

        # Step 3: Top-down routing (from max_level down to new_level + 1)
        for l in range(curr_level, new_level, -1):
            search_res = self._search_layer(vector, curr_ep, ef=1, layer=l)
            curr_ep = [search_res[0][1]]  # Keep single closest entry point

        # Step 4: Multi-layer insertion and neighbor linkage (from min(new_level, max_level) down to 0)
        for l in range(min(new_level, curr_level), -1, -1):
            # Find candidate neighbors at layer l using ef_construction
            candidates = self._search_layer(vector, curr_ep, ef=self.ef_construction, layer=l)

            # Max allowed connections for this layer
            max_conn = self.Mmax0 if l == 0 else self.Mmax

            # Select best neighbors
            neighbors = self._select_neighbors(candidates, self.M)

            # Establish bidirectional links
            if node_id not in self.layers[l]:
                self.layers[l][node_id] = set()

            for neighbor in neighbors:
                self.layers[l][node_id].add(neighbor)

                if neighbor not in self.layers[l]:
                    self.layers[l][neighbor] = set()
                self.layers[l][neighbor].add(node_id)

                # Prune neighbor connections if exceeding layer capacity limit
                if len(self.layers[l][neighbor]) > max_conn:
                    # Re-score neighbor's connections relative to neighbor vector
                    n_vec = self.vectors[neighbor]
                    curr_conns = list(self.layers[l][neighbor])
                    conn_candidates = [(self._distance(n_vec, self.vectors[c]), c) for c in curr_conns]

                    pruned_neighbors = self._select_neighbors(conn_candidates, max_conn)
                    self.layers[l][neighbor] = set(pruned_neighbors)

            # Update entry points for next lower layer search
            curr_ep = [node for _, node in candidates]

        # Step 5: Update global entry point if new_level exceeds max_level
        if new_level > self.max_level:
            self.max_level = new_level
            self.entry_point = node_id

    def search(self, query: np.ndarray, k: int = 5, ef_search: int = 50) -> list[int]:
        """Full query procedure (Algorithm 5 in paper)."""
        if self.entry_point is None:
            return []

        curr_ep = [self.entry_point]

        # Step 1: Routing down to layer 1
        for l in range(self.max_level, 0, -1):
            search_res = self._search_layer(query, curr_ep, ef=1, layer=l)
            curr_ep = [search_res[0][1]]

        # Step 2: Layer 0 search with ef_search budget
        search_res = self._search_layer(query, curr_ep, ef=max(ef_search, k), layer=0)

        # Step 3: Return top-k closest node IDs
        return [node for _, node in search_res[:k]]