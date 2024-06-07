import torch
from typing import Generator

from rsl_rl.storage.replay_storage import ReplayStorage
from rsl_rl.storage.storage import Dataset, Transition


class PPGStorage(ReplayStorage):
    """Implementation storage(buffer) in PPG algorithm. Different from RolloutStorage, PPGStorage would save data of n_pi interations."""

    def __init__(self, environment_count: int, device: str = "cpu", n_pi: int = 32):
        """
        Args:
            environment_count (int): Number of environments.
            device (str, optional): Device to use. Defaults to "cpu".
            n_pi (int): n_pi in PPG algorithm. The buffer will keep n_pi interations of data.
        """
        super().__init__(environment_count, environment_count, device=device, initial_size=0)
        self._size_initialized = False
        self._n_pi = n_pi
        

    def append(self, dataset: Dataset) -> None:
        """Appends a dataset to the ppg storage.

        Args:
            dataset (Dataset): Dataset to append.
        Raises:
            AssertionError: If the dataset is not of the correct size.
        """
        if not self._size_initialized:
            self.max_size = len(dataset) * self._env_count * self._n_pi
            self._size_initialized = True

        assert self._size == len(dataset) * self._n_pi

        super().append(dataset)

    @property
    def idx(self) -> int:
        return self._idx


    def batch_generator(self, batch_count: int, trajectories: bool = False) -> Generator[Transition, None, None]:
        """Yields batches of transitions or trajectories.

        Args:
            batch_count (int): Number of batches to yield.
            trajectories (bool, optional): Whether to yield batches of trajectories. Defaults to False.

        Returns:
            Generator yielding batches of transitions of shape (batch_size, *shape). If trajectories is True, yields
            batches of trajectories of shape (env_count, steps_per_env, *shape).
        """
        assert self._initialized, "PPG storage must be initialized."

        total_size = self._env_count if trajectories else self._size * self._env_count // self._n_pi
        batch_size = total_size // batch_count
        indices = torch.randperm(total_size)

        assert batch_size > 0, "Batch count is too large."

        if trajectories:
            # Reshape to (env_count, steps_per_env, *shape)
            data = {k: v.reshape(-1, self._env_count, *v.shape[1:]).transpose(0, 1) for k, v in self._data.items()}
        else:
            data = self._data

        start_idx = (self._idx - self._size // self._n_pi) * self._env_count if self._idx > 0 else (self._n_pi - 1) * self._size // self._n_pi * self._env_count
        indices += start_idx
        for i in range(batch_count):
            batch_idx = indices[i * batch_size : (i + 1) * batch_size].detach().to(self.device)

            batch = {}
            for key, value in data.items():
                batch[key] = self._process_undo(key, value[batch_idx].clone())

            yield batch

    def overall_batch_generator(self, batch_count: int, trajectories: bool = False) -> Generator[Transition, None, None]:
        """Yield batches of data from all data. This is designed for PPG algorithm's auxiliary stage

        """
        total_size = self._env_count if trajectories else self._size * self._env_count
        batch_size = total_size // batch_count
        indices = torch.randperm(total_size)

        if trajectories:
            # Reshape to (env_count, steps_per_env, *shape)
            data = {k: v.reshape(-1, self._env_count, *v.shape[1:]).transpose(0, 1) for k, v in self._data.items()}
        else:
            data = self._data

        for i in range(batch_count):
            batch_idx = indices[i * batch_size : (i + 1) * batch_size].detach().to(self.device)

            batch = {}
            for key, value in data.items():
                batch[key] = self._process_undo(key, value[batch_idx].clone())

            yield batch
