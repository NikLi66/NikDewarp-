import zipfile
import io
import numpy as np
def zip_encode(array):
    # 将 numpy 数组转换为字节数据
    byte_data = array.tobytes()

    # 使用 zip 压缩字节数据
    compressed_data = io.BytesIO()
    with zipfile.ZipFile(compressed_data, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('array.npy', byte_data)

    # 获取压缩后的数据
    compressed_data.seek(0)
    compressed_bytes = compressed_data.read()

    return compressed_bytes

def zip_decode(bytes, dtype, shape):
    # 解压缩并还原 numpy 数组
    decompressed_data = io.BytesIO(bytes)
    with zipfile.ZipFile(decompressed_data, 'r') as zf:
        with zf.open('array.npy') as file:
            byte_data = file.read()
            array = np.frombuffer(byte_data, dtype=dtype).reshape(shape)

    return array