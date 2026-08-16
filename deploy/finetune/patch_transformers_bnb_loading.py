from pathlib import Path

path = Path("/usr/local/lib/python3.12/dist-packages/transformers/core_model_loading.py")
source = path.read_text(encoding="utf-8")
old = """            if future_or_tensor is None:
                param_device = get_device(device_map, renamed_key, valid_torch_device=True)
                future_or_tensor = spawn_materialize(thread_pool, tensor, param_device, _dtype)
"""
new = """            if future_or_tensor is None:
                # Transformers 5.2 otherwise materializes BNB-bound tensors on the GPU
                # in BF16 before quantizing them, which OOMs Qwen3.6-27B on a 48 GiB GPU.
                # Quantization operations consume CPU tensors and place the converted result.
                param_device = (
                    \"cpu\"
                    if mapping.quantization_operation is not None
                    else get_device(device_map, renamed_key, valid_torch_device=True)
                )
                future_or_tensor = spawn_materialize(thread_pool, tensor, param_device, _dtype)
"""
if new in source:
    print("transformers BNB materialization patch already present")
elif old not in source:
    raise SystemExit("transformers BNB materialization patch target not found")
else:
    path.write_text(source.replace(old, new, 1), encoding="utf-8")
    print("patched transformers BNB materialization to quantize from CPU")
