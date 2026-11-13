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

import pathlib

from absl.testing import absltest
from absl.testing import flagsaver
from absl.testing import parameterized
import numpy as np
from PIL import Image

import os
import io
from litert_quantizer import calibrator
from litert_quantizer.algorithms.uniform_quantize import naive_min_max_quantize
from litert_quantizer.examples.selfie_segmentation import quantize_selfie_segmentation
from litert_quantizer.utils import test_utils

RangeMethod = naive_min_max_quantize.ActivationRangeMethod
CalibrationMode = calibrator.CalibrationMode

# Older ai-edge-litert packages (e.g. in OSS CI) can't collect histograms in
# profiler-based mode.
_HAS_PROFILER_HISTOGRAM_SUPPORT = hasattr(
    calibrator._pywrap_tfl_calibration, 'TensorHistogram'  # pylint: disable=protected-access
)


def _write_synthetic_images(image_dir: str, num_images: int = 2) -> list[str]:
  """Writes random RGB JPEG images and returns their paths."""
  rng = np.random.default_rng(0)
  paths = []
  for i in range(num_images):
    pixels = rng.integers(0, 256, size=(40, 48, 3), dtype=np.uint8)
    path = str(pathlib.Path(image_dir) / f'image_{i}.jpg')
    Image.fromarray(pixels).save(path)
    paths.append(path)
  return paths


class QuantizeSelfieSegmentationTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._model_path = str(
        pathlib.Path(test_utils.get_path_to_datafile('../../tests/models'))
        / 'conv_fc_mnist.tflite'
    )
    self._image_paths = _write_synthetic_images(self.create_tempdir().full_path)

  def test_load_and_preprocess_images_produces_expected_shapes(self):
    input_name, input_shape, _ = (
        quantize_selfie_segmentation.get_model_input_spec(self._model_path)
    )
    self.assertNotEmpty(input_name)
    calib_samples, eval_samples = (
        quantize_selfie_segmentation.load_and_preprocess_images(
            self._image_paths,
            target_shape=input_shape,
            num_augmented_samples=10,
        )
    )
    self.assertLen(calib_samples, 10)
    self.assertLen(eval_samples, len(self._image_paths))
    for arr in calib_samples:
      self.assertEqual(arr.shape, input_shape)
      self.assertEqual(arr.dtype, np.float32)
      self.assertGreaterEqual(float(np.min(arr)), 0.0)
      self.assertLessEqual(float(np.max(arr)), 1.0)

  @parameterized.named_parameters(
      ('min_max_profiler', RangeMethod.MIN_MAX),
      ('histogram_percentile_profiler', RangeMethod.HISTOGRAM_PERCENTILE),
      ('histogram_mse_profiler', RangeMethod.HISTOGRAM_MSE),
  )
  def test_calibrate_quantize_and_evaluate_a8w8(self, range_method):
    if (
        range_method != RangeMethod.MIN_MAX
        and not _HAS_PROFILER_HISTOGRAM_SUPPORT
    ):
      self.skipTest('Requires _pywrap_tfl_calibration.TensorHistogram')
    input_name, input_shape, _ = (
        quantize_selfie_segmentation.get_model_input_spec(self._model_path)
    )
    calib_samples, eval_samples = (
        quantize_selfie_segmentation.load_and_preprocess_images(
            self._image_paths,
            target_shape=input_shape,
            num_augmented_samples=8,
        )
    )
    calib_dataset = quantize_selfie_segmentation.build_calibration_dataset(
        input_name, calib_samples
    )
    float_outputs = quantize_selfie_segmentation.run_model_on_samples(
        self._model_path, eval_samples
    )
    out_dir = self.create_tempdir().full_path

    quant_bytes, report = (
        quantize_selfie_segmentation.calibrate_and_quantize_a8w8(
            float_model_path=self._model_path,
            calibration_data=calib_dataset,
            eval_samples=eval_samples,
            float_outputs=float_outputs,
            range_method=range_method,
            calibration_mode=CalibrationMode.CALIBRATION_PROFILER_BASED,
            output_dir=out_dir,
            model_name=f'test_{range_method.value}',
        )
    )
    self.assertNotEmpty(quant_bytes)
    self.assertGreater(report.compression_ratio, 1.0)
    self.assertGreater(report.cosine_similarity, 0.90)
    self.assertGreaterEqual(report.calibration_time_sec, 0.0)

  def test_evaluate_identical_model_gives_perfect_agreement(self):
    _, input_shape, _ = quantize_selfie_segmentation.get_model_input_spec(
        self._model_path
    )
    _, eval_samples = quantize_selfie_segmentation.load_and_preprocess_images(
        self._image_paths, target_shape=input_shape, num_augmented_samples=6
    )
    float_outputs = quantize_selfie_segmentation.run_model_on_samples(
        self._model_path, eval_samples
    )
    with open(self._model_path, 'rb') as f:
      model_bytes = f.read()
    report = quantize_selfie_segmentation.evaluate_segmentation_quality(
        float_outputs=float_outputs,
        quant_model_bytes=model_bytes,
        eval_samples=eval_samples,
        model_name='identity',
        range_method='none',
        calibration_mode=CalibrationMode.CALIBRATION_PROFILER_BASED,
        calibration_time_sec=0.0,
        float_model_size_bytes=len(model_bytes),
    )
    self.assertEqual(report.output_mse, 0.0)
    self.assertAlmostEqual(report.cosine_similarity, 1.0)
    self.assertAlmostEqual(report.kl_divergence, 0.0)
    self.assertEqual(report.pixel_top1_agreement, 1.0)
    self.assertEqual(report.mean_iou, 1.0)
    self.assertNotEmpty(report.per_class_iou)
    self.assertTrue(all(v == 1.0 for v in report.per_class_iou.values()))
    self.assertEqual(report.compression_ratio, 1.0)

  @absltest.skipIf(
      not _HAS_PROFILER_HISTOGRAM_SUPPORT,
      'Requires _pywrap_tfl_calibration.TensorHistogram',
  )
  def test_main_cli_execution_with_flags(self):
    out_dir = self.create_tempdir().full_path
    out_tflite = str(pathlib.Path(out_dir) / 'model_mse.tflite')
    with flagsaver.flagsaver(
        float_model_path=self._model_path,
        image_paths=self._image_paths,
        range_method='histogram_mse',
        calibration_method='CALIBRATION_PROFILER_BASED',
        output_tflite_path=out_tflite,
        num_calib_samples=4,
        output_dir=out_dir,
    ):
      quantize_selfie_segmentation.main(['quantize_selfie_segmentation.py'])

    self.assertTrue(pathlib.Path(out_tflite).exists())
    self.assertGreater(pathlib.Path(out_tflite).stat().st_size, 0)


if __name__ == '__main__':
  absltest.main()
