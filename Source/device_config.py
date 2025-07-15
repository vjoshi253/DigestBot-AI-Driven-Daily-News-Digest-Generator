import logging
import torch
import log_config  # Ensure logging is configured

logger = logging.getLogger(__name__)


class DeviceManager:
    """
    Manages the torch device selection and ensures the device type is logged only once.
    """

    _device = None
    _device_string = None

    @classmethod
    def override_device(cls) -> None:
        """
        Overrides the device to CPU.
        """
        cls._device = torch.device("cpu")
        cls._device_string = "cpu"
        logger.info("Overriding device to CPU")

    @classmethod
    def get_torch_device(cls) -> torch.device:
        """
        Lazily determines and returns the best available torch.device.
        Logs the selected device only on first access.

        Returns:
            torch.device: One of 'mps', 'cuda', or 'cpu'.
        """
        if cls._device is not None:
            return cls._device

        if torch.backends.mps.is_available():
            cls._device = torch.device("mps")
            logger.info("Using MPS device")
        elif torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            cls._device = torch.device("cuda")
            logger.info(f"Using CUDA device: {device_name}")
        else:
            cls._device = torch.device("cpu")
            logger.info("Using CPU device")

        return cls._device

    @classmethod
    def get_torch_type(cls) -> str:
        """
        Lazily determines and returns the device string.
        Logs the selected string only on first access.

        Returns:
            str: One of 'mps', 'cuda', or 'cpu'.
        """
        if cls._device_string is not None:
            return cls._device_string

        # Call get_device to ensure logging is consistent
        device = cls.get_torch_device()
        cls._device_string = device.type
        return cls._device_string
