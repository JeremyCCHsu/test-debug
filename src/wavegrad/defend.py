from typing import Optional, Tuple, List

import numpy as np
import torch

from art.config import ART_NUMPY_DTYPE
from art.defences.preprocessor import Preprocessor

from wavegrad.denoise import Denoiser
from wavegrad.preprocess import extract_melspectrogram


device = "cuda" if torch.cuda.is_available() else "cpu"


class WaveGradDenoiser(Preprocessor):
    """
    Abstract base class for preprocessing defences.
    """

    params: List[str] = []

    def __init__(
        self,
        batch_size,
        denoiser_dir,
        severity,
        is_fitted=True,
        # clip_values=[-1., 1.],
    ) -> None:
        """
        Create a preprocessing object.
        """
        self._is_fitted = is_fitted
        # self._apply_fit = apply_fit
        # self._apply_predict = apply_predict
        self.clip_values = [-1., 1.]
        # self.eps = eps
        self.batch_size = batch_size
        # self.wavegrad = wavegrad
        # self.verbose = verbose

        self.denoiser = Denoiser(denoiser_dir, params=dict(), device=device)
        self.severity = severity

        self._check_params()  # TODO

    def __call__(
        self, 
        x: np.ndarray, 
        y: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Perform data preprocessing and return preprocessed data as tuple.
        :param x: Dataset to be preprocessed.
        :param y: Labels to be preprocessed.
        :return: Preprocessed data.
        """
        x_copy = x + 0
        original_shape = x.shape

        # ============
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).to(device)

        if len(original_shape) == 3:
            x = x.squeeze(1)

        with torch.no_grad():
            # NOTE BUG this could throw error in torch==1.6.0
            mel = extract_melspectrogram(x.cpu())

            # FIXME: why does it always get me an extra?
            mel = mel[..., :-1].to(device)  
            x_denoised = (
                self.denoiser(mel, x, self.severity)
                .clamp(*self.clip_values) 
                .to("cpu")
                .numpy()
            )

        x_denoised = x_denoised.astype(ART_NUMPY_DTYPE)
        
        new_sequence_len = x_denoised.shape[-1]
        x_copy[..., :new_sequence_len] = x_denoised  # HACK
        return (x_copy, y)

    def estimate_gradient(self, x: np.ndarray, grad: np.ndarray) -> np.ndarray:
        """
        Provide an estimate of the gradients of the defence for the backward pass. If the defence is not differentiable,
        this is an estimate of the gradient, most often replacing the computation performed by the defence with the
        identity function.
        :param x: Input data for which the gradient is estimated. First dimension is the batch size.
        :param grad: Gradient value so far.
        :return: The gradient (estimate) of the defence.
        """
        return grad

    def _check_params(self) -> None:
        if self.denoiser is None:
            raise

    @property
    def apply_predict(self) -> bool:
        return self._apply_predict
    
    @property
    def apply_fit(self) -> bool:
        return self._apply_fit
    
    def fit(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> None:
        """
        No parameters to learn for this method; do nothing.
        """
        pass
