# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Copyright 2026 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""End-to-end histogram calibration and A8W8 static quantization example."""

from collections.abc import Mapping, Sequence
import dataclasses
import math
import pathlib
import time
from typing import Any
import urllib.request

from absl import app
from absl import flags
import numpy as np
from PIL import Image

import os
import io
from litert_quantizer import calibrator
from litert_quantizer import quantizer
from litert_quantizer import recipe
from litert_quantizer.algorithms.uniform_quantize import naive_min_max_quantize
from litert_quantizer.utils import test_utils
from litert_quantizer.utils import tfl_interpreter_utils

RangeMethod = naive_min_max_quantize.ActivationRangeMethod
CalibrationMode = calibrator.CalibrationMode

SELFIE_CLASS_NAMES = (
    'background',
    'hair',
    'body_skin',
    'face_skin',
    'clothes',
    'others',
)

_CALIBRATION_MODES = {
    'CALIBRATION_PROFILER_BASED': CalibrationMode.CALIBRATION_PROFILER_BASED,
    'CALIBRATION_PRESERVE_ALL_TENSORS': (
        CalibrationMode.CALIBRATION_PRESERVE_ALL_TENSORS
    ),
}

_ASYNC_SEGMENTATION_DIR = '../../../../litert/litert/samples/async_segmentation'
_DEFAULT_MODEL_PATH = test_utils.get_path_to_datafile(
    f'{_ASYNC_SEGMENTATION_DIR}/models/selfie_multiclass_256x256.tflite'
)
# Test image from the LiteRT async segmentation sample.
DEFAULT_IMAGE_PATH = test_utils.get_path_to_datafile(
    f'{_ASYNC_SEGMENTATION_DIR}/test_images/image.jpeg'
)
_SELFIE_MODEL_GCS_URL = (
    'https://storage.googleapis.com/mediapipe-models/image_segmenter/'
    'selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite'
)

_FLOAT_MODEL_PATH = flags.DEFINE_string(
    'float_model_path',
    _DEFAULT_MODEL_PATH,
    'Path to the floating-point TFLite segmentation model.',
)
_IMAGE_PATHS = flags.DEFINE_list(
    'image_paths',
    [DEFAULT_IMAGE_PATH],
    'Comma-separated calibration and evaluation image paths.',
)
_OUTPUT_DIR = flags.DEFINE_string(
    'output_dir',
    '/tmp/selfie_a8w8_quantization',
    'Directory to save quantized models and recipes.',
)
_NUM_CALIB_SAMPLES = flags.DEFINE_integer(
    'num_calib_samples',
    32,
    'Number of augmented image samples to use for calibration.',
)
_RANGE_METHOD = flags.DEFINE_enum(
    'range_method',
    None,
    [m.value for m in RangeMethod],
    'Optional activation range method to run. If None, runs all four default'
    ' configurations.',
)
_CALIBRATION_METHOD = flags.DEFINE_enum(
    'calibration_method',
    'CALIBRATION_PROFILER_BASED',
    list(_CALIBRATION_MODES.keys()),
    'Calibration method used when --range_method is specified.',
)
_OUTPUT_TFLITE_PATH = flags.DEFINE_string(
    'output_tflite_path',
    None,
    'Optional path where the single quantized TFLite model should be written.',
)


@dataclasses.dataclass(frozen=True)
class SegmentationQualityReport:
  """Numerical and task-level quality metrics for a quantized model."""

  model_name: str
  range_method: str
  calibration_mode: CalibrationMode
  calibration_time_sec: float
  model_size_mb: float
  compression_ratio: float
  output_mse: float
  output_sqnr_db: float
  cosine_similarity: float
  kl_divergence: float
  pixel_top1_agreement: float
  mean_iou: float
  per_class_iou: dict[str, float]


def get_model_input_spec(model_path: str) -> tuple[str, tuple[int, ...], Any]:
  """Returns `(input_name, input_shape, input_dtype)` for the default signature."""
  interp = tfl_interpreter_utils.create_tfl_interpreter(
      model_path, allocate_tensors=False
  )
  sig_defs = interp.get_signature_list()
  input_details = interp.get_input_details()
  if sig_defs and tfl_interpreter_utils.DEFAULT_SIGNATURE_KEY in sig_defs:
    sig_inputs = sig_defs[tfl_interpreter_utils.DEFAULT_SIGNATURE_KEY]['inputs']
    input_name = sig_inputs[0]
  else:
    input_name = input_details[0]['name'].split(':')[0]
  shape = tuple(int(d) for d in input_details[0]['shape'])
  return input_name, shape, input_details[0]['dtype']


def load_and_preprocess_images(
    image_paths: Sequence[str],
    target_shape: tuple[int, ...],
    num_augmented_samples: int = 32,
    seed: int = 42,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
  """Loads images and generates `(calibration_arrays, eval_arrays)`.

  The evaluation arrays are the resized images. The calibration arrays are the
  same images plus random crops, flips, and contrast/brightness changes of them,
  up to `num_augmented_samples` arrays.

  Args:
    image_paths: Paths of the images to load.
    target_shape: The model input shape, `(1, height, width, channels)`.
    num_augmented_samples: Number of calibration arrays to return.
    seed: Seed for the random augmentations.

  Returns:
    A tuple of the calibration arrays and the evaluation arrays, each of shape
    `target_shape` with values in [0, 1].
  """
  if not image_paths:
    raise ValueError('No image paths provided.')
  _, height, width, channels = target_shape
  base_images: list[np.ndarray] = []
  eval_arrays: list[np.ndarray] = []
  mode = 'L' if channels == 1 else 'RGB'

  for img_path in image_paths:
    with open(img_path, 'rb') as f:
      pil_img = Image.open(f).convert(mode)
      base_images.append(np.asarray(pil_img, dtype=np.float32) / 255.0)
      resized = pil_img.resize((width, height), Image.Resampling.BILINEAR)
      arr = np.asarray(resized, dtype=np.float32) / 255.0
      eval_arrays.append(arr.reshape((1, height, width, channels)))

  calib_arrays: list[np.ndarray] = list(eval_arrays)
  rng = np.random.default_rng(seed)

  while len(calib_arrays) < num_augmented_samples:
    src = base_images[len(calib_arrays) % len(base_images)]
    src_h, src_w = src.shape[0], src.shape[1]

    crop_scale = float(rng.uniform(0.70, 1.0))
    crop_h = max(1, int(src_h * crop_scale))
    crop_w = max(1, int(src_w * crop_scale))
    top = int(rng.integers(0, max(1, src_h - crop_h + 1)))
    left = int(rng.integers(0, max(1, src_w - crop_w + 1)))
    cropped = src[top : top + crop_h, left : left + crop_w]

    uint8_crop = np.clip(cropped * 255.0, 0.0, 255.0).astype(np.uint8)
    pil_crop = Image.fromarray(uint8_crop, mode=mode).resize(
        (width, height), Image.Resampling.BILINEAR
    )
    aug = np.asarray(pil_crop, dtype=np.float32) / 255.0
    if rng.random() < 0.5:
      aug = np.flip(aug, axis=1)

    contrast = float(rng.uniform(0.85, 1.15))
    brightness = float(rng.uniform(-0.08, 0.08))
    mean_val = float(np.mean(aug))
    aug = np.clip((aug - mean_val) * contrast + mean_val + brightness, 0.0, 1.0)
    calib_arrays.append(
        aug.astype(np.float32).reshape((1, height, width, channels))
    )

  return calib_arrays[:num_augmented_samples], eval_arrays


def build_calibration_dataset(
    input_name: str, samples: Sequence[np.ndarray]
) -> dict[str, list[dict[str, np.ndarray]]]:
  """Formats input arrays into an AEQ signature-keyed calibration dataset."""
  return {
      tfl_interpreter_utils.DEFAULT_SIGNATURE_KEY: [
          {input_name: sample} for sample in samples
      ]
  }


def run_model_on_samples(
    model_source: str | bytes, samples: Sequence[np.ndarray]
) -> list[np.ndarray]:
  """Runs TFLite inference on `samples` and returns dequantized float outputs."""
  interp = tfl_interpreter_utils.create_tfl_interpreter(model_source)
  output_details = interp.get_output_details()[0]

  outputs: list[np.ndarray] = []
  for sample in samples:
    tfl_interpreter_utils.invoke_interpreter_once(interp, [sample])
    out = tfl_interpreter_utils.get_tensor_data(
        interp, output_details, dequantize=True
    )
    outputs.append(np.asarray(out, dtype=np.float32).copy())
  return outputs


def evaluate_segmentation_quality(
    float_outputs: Sequence[np.ndarray],
    quant_model_bytes: bytes,
    eval_samples: Sequence[np.ndarray],
    model_name: str,
    range_method: str,
    calibration_mode: CalibrationMode,
    calibration_time_sec: float,
    float_model_size_bytes: int,
) -> SegmentationQualityReport:
  """Evaluates numerical and segmentation quality of a quantized model."""
  quant_outputs = run_model_on_samples(quant_model_bytes, eval_samples)

  ref_all = np.concatenate(float_outputs, axis=0).astype(np.float64)
  q_all = np.concatenate(quant_outputs, axis=0).astype(np.float64)
  diff = q_all - ref_all

  mse = float(np.mean(np.square(diff)))
  signal_power = float(np.mean(np.square(ref_all)))
  sqnr_db = 10.0 * math.log10(signal_power / max(mse, 1e-15))

  ref_flat = ref_all.ravel()
  q_flat = q_all.ravel()
  denom = float(np.linalg.norm(ref_flat) * np.linalg.norm(q_flat))
  cosine_sim = float(np.dot(ref_flat, q_flat) / max(denom, 1e-15))

  eps = 1e-7

  def softmax(logits: np.ndarray) -> np.ndarray:
    exp = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return np.clip(exp / np.sum(exp, axis=-1, keepdims=True), eps, 1.0)

  p_ref = softmax(ref_all)
  p_q = softmax(q_all)
  kl_div = float(np.mean(np.sum(p_ref * np.log(p_ref / p_q), axis=-1)))

  ref_mask = np.argmax(ref_all, axis=-1)
  q_mask = np.argmax(q_all, axis=-1)
  pixel_agreement = float(np.mean(ref_mask == q_mask))

  per_class_iou: dict[str, float] = {}
  for c in range(ref_all.shape[-1]):
    class_name = (
        SELFIE_CLASS_NAMES[c] if c < len(SELFIE_CLASS_NAMES) else f'class_{c}'
    )
    ref_c = ref_mask == c
    q_c = q_mask == c
    union = int(np.sum(ref_c | q_c))
    if union == 0:
      continue
    per_class_iou[class_name] = float(np.sum(ref_c & q_c)) / float(union)
  mean_iou = float(np.mean(list(per_class_iou.values())))

  quant_size = len(quant_model_bytes)
  return SegmentationQualityReport(
      model_name=model_name,
      range_method=range_method,
      calibration_mode=calibration_mode,
      calibration_time_sec=calibration_time_sec,
      model_size_mb=quant_size / (1024.0 * 1024.0),
      compression_ratio=float(float_model_size_bytes) / max(quant_size, 1),
      output_mse=mse,
      output_sqnr_db=sqnr_db,
      cosine_similarity=cosine_sim,
      kl_divergence=kl_div,
      pixel_top1_agreement=pixel_agreement,
      mean_iou=mean_iou,
      per_class_iou=per_class_iou,
  )


def calibrate_and_quantize_a8w8(
    float_model_path: str,
    calibration_data: Mapping[str, Sequence[Mapping[str, np.ndarray]]],
    eval_samples: Sequence[np.ndarray],
    float_outputs: Sequence[np.ndarray],
    range_method: RangeMethod,
    calibration_mode: CalibrationMode,
    output_dir: str,
    model_name: str,
) -> tuple[bytes, SegmentationQualityReport]:
  """Calibrates, quantizes to A8W8, and evaluates quality."""
  qt = quantizer.Quantizer(float_model_path)
  qt.load_quantization_recipe(
      recipe.static_wi8_ai8(
          activation_algorithm_params={'range_method': range_method.value}
      )
  )

  t0 = time.perf_counter()
  calibration_result = qt.calibrate(
      calibration_data,  # pyrefly: ignore[bad-argument-type]
      mode=calibration_mode,
  )
  calibration_time = time.perf_counter() - t0

  quant_result = qt.quantize(calibration_result)
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
  quant_result.save(output_dir, model_name=model_name, overwrite=True)

  raw_bytes = quant_result.quantized_model
  quant_bytes = (
      bytes(raw_bytes) if isinstance(raw_bytes, bytearray) else raw_bytes
  )
  assert quant_bytes is not None

  float_size = os.path.getsize(float_model_path)
  quality_report = evaluate_segmentation_quality(
      float_outputs=float_outputs,
      quant_model_bytes=quant_bytes,
      eval_samples=eval_samples,
      model_name=model_name,
      range_method=range_method.value,
      calibration_mode=calibration_mode,
      calibration_time_sec=calibration_time,
      float_model_size_bytes=float_size,
  )
  return quant_bytes, quality_report


def print_quality_reports(
    reports: Sequence[SegmentationQualityReport],
) -> None:
  """Prints a formatted segmentation quality table."""
  print(
      f'\n{"Model / Algorithm":<36} {"Mode":<10} {"Calib(s)":>8} '
      f'{"Size(MB)":>8} {"SQNR(dB)":>9} {"MSE":>10} {"CosSim":>8} '
      f'{"KL Div":>9} {"Top1 Acc":>9} {"mIoU":>8}'
  )
  for r in reports:
    mode_short = (
        'PROFILER'
        if r.calibration_mode == CalibrationMode.CALIBRATION_PROFILER_BASED
        else 'PRESERVE'
    )
    print(
        f'{r.model_name:<36} {mode_short:<10} '
        f'{r.calibration_time_sec:>8.1f} {r.model_size_mb:>8.2f} '
        f'{r.output_sqnr_db:>9.2f} {r.output_mse:>10.2e} '
        f'{r.cosine_similarity:>8.5f} {r.kl_divergence:>9.5f} '
        f'{100.0 * r.pixel_top1_agreement:>8.2f}% '
        f'{100.0 * r.mean_iou:>7.2f}%'
    )


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  model_path = _FLOAT_MODEL_PATH.value
  output_dir = _OUTPUT_DIR.value
  assert model_path is not None
  assert output_dir is not None

  if not os.path.exists(model_path):
    if model_path == _DEFAULT_MODEL_PATH:
      if not os.path.exists(output_dir):
        os.makedirs(output_dir)
      downloaded_path = str(
          pathlib.Path(output_dir) / 'selfie_multiclass_256x256.tflite'
      )
      if not os.path.exists(downloaded_path):
        print(f'Downloading selfie model from {_SELFIE_MODEL_GCS_URL} ...')
        urllib.request.urlretrieve(_SELFIE_MODEL_GCS_URL, downloaded_path)
      model_path = downloaded_path
    else:
      raise ValueError(f'Model file does not exist: {model_path}')

  input_name, input_shape, _ = get_model_input_spec(model_path)
  calib_samples, eval_samples = load_and_preprocess_images(
      _IMAGE_PATHS.value,
      target_shape=input_shape,
      num_augmented_samples=_NUM_CALIB_SAMPLES.value,
  )
  calib_dataset = build_calibration_dataset(input_name, calib_samples)
  float_outputs = run_model_on_samples(model_path, eval_samples)

  if _RANGE_METHOD.value is not None:
    method = RangeMethod(_RANGE_METHOD.value)
    mode = _CALIBRATION_MODES[_CALIBRATION_METHOD.value]
    configs = [(f'selfie_a8w8_{method.value}', method, mode)]
  else:
    configs = [
        (
            'selfie_a8w8_minmax_profiler',
            RangeMethod.MIN_MAX,
            CalibrationMode.CALIBRATION_PROFILER_BASED,
        ),
        (
            'selfie_a8w8_hist_pct_profiler',
            RangeMethod.HISTOGRAM_PERCENTILE,
            CalibrationMode.CALIBRATION_PROFILER_BASED,
        ),
        (
            'selfie_a8w8_hist_mse_profiler',
            RangeMethod.HISTOGRAM_MSE,
            CalibrationMode.CALIBRATION_PROFILER_BASED,
        ),
        (
            'selfie_a8w8_hist_mse_preserve',
            RangeMethod.HISTOGRAM_MSE,
            CalibrationMode.CALIBRATION_PRESERVE_ALL_TENSORS,
        ),
    ]

  reports: list[SegmentationQualityReport] = []
  for name, method, mode in configs:
    quant_bytes, report = calibrate_and_quantize_a8w8(
        float_model_path=model_path,
        calibration_data=calib_dataset,
        eval_samples=eval_samples,
        float_outputs=float_outputs,
        range_method=method,
        calibration_mode=mode,
        output_dir=output_dir,
        model_name=name,
    )
    reports.append(report)
    if _OUTPUT_TFLITE_PATH.value is not None and len(configs) == 1:
      out_tflite = pathlib.Path(_OUTPUT_TFLITE_PATH.value)
      if not os.path.exists(str(out_tflite.parent)):
        os.makedirs(str(out_tflite.parent))
      with open(str(out_tflite), 'wb') as f:
        f.write(quant_bytes)

  print_quality_reports(reports)


if __name__ == '__main__':
  app.run(main)
