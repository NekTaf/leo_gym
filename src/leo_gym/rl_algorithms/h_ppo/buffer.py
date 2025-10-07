# Third-party
import numpy as np
import torch as T


class TrajectoryBuffer:
    """
    Each 'push' call should supply a dict of named arrays/tensors for a single timestep.
    Internally, we keep a list of Python dicts; when you call `generate_batches()`,
    we stack each field into a NumPy array and return (all_data_dict, list_of_index_batches).
    """

    def __init__(self, batch_size: int):
        self.batch_size = batch_size
        self.storage = []  
        self._keys = None
        

    def push(self, 
             data: dict
             )->None:
        
        if self._keys is None:
            self._keys = set(data.keys())
        else:
            if set(data.keys()) != self._keys:
                missing = self._keys - set(data.keys())
                extra   = set(data.keys()) - self._keys
                raise ValueError(
                    f"TrajectoryBuffer.push: field-keys mismatch. "
                    f"Missing keys: {missing}, extra keys: {extra}"
                )
        step_data = {}
        for k, v in data.items():
            if isinstance(v, T.Tensor):
                step_data[k] = v.detach().cpu().numpy()
            else:
                step_data[k] = v
        self.storage.append(step_data)


    def generate_batches(self):
        """
        Convert the stored list-of-dicts into a dict-of-arrays, and produce minibatch indices.

        Returns:
            all_data: dict mapping each field name → NumPy array of shape (T, *field_shape)
            batches: a list of NumPy 1D index‐arrays, each of length `batch_size` (except possibly the last).
        """
        if not self.storage:
            raise ValueError("TrajectoryBuffer.generate_batches: buffer is empty")

        n_steps = len(self.storage)
        all_data = {}
        
        for key in self._keys:
            # if first_val is scalar -> (n_steps,)
            # if an array (n_envs,) or (n_envs, dim) -> (n_steps, n_envs, dim)
            arr_list = [step[key] for step in self.storage]
            all_data[key] = np.stack(arr_list, axis=0)

        indices = np.arange(n_steps, dtype=np.int64)
        np.random.shuffle(indices)
        
        batch_starts = np.arange(0, n_steps, self.batch_size)
        batches = [indices[i : i + self.batch_size] for i in batch_starts]

        return all_data, batches


    def clear(self):
        self.storage = []
        self._keys = None
