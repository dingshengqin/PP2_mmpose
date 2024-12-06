_base_ = ['./pose-detection_static.py', '../_base_/backends/onnxruntime.py']

onnx_config = dict(
    input_shape=[416, 416],
    output_names=['simcc_x', 'simcc_y'],
    dynamic_axes={
        'input': {
            0: 'batch',
        },
        'simcc_x': {
            0: 'batch'
        },
        'simcc_y': {
            0: 'batch'
        }
    })
