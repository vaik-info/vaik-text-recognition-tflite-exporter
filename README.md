# vaik-text-recognition-tflite-exporter

Export from OCR pb model to tflite model

## Usage

```shell
pip install -r requirements.txt
python export_tflite.py --input_model_dir_path ~/.vaik_text_recognition_pb_trainer/output_model/2023-02-10-07-10-32/step-5000_batch-16_epoch-113_loss_0.1720_val_loss_0.8170_train_20h_exscore_0.88_tcn \
                --output_model_file_path ~/.vaik_text_recognition_pb_exporter/model.tflite \
                --input_height 96 \
                --input_width 576
```

## Output

- ```~/.vaik_text_recognition_pb_exporter/model.tflite```