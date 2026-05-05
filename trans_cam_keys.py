import pickle, os, sys
import numpy as np
old_cache = '/data/sunzhihao2/turuirui/arctic_extrinsics/arctic_camera_cache_with_extrinsics.pkl'
new_cache = '/data/sunzhihao2/turuirui/arctic_extrinsics/arctic_camera_cache_with_extrinsics_local.pkl'
# 训练脚本中使用的 dataset_folder（如需替换，请修改下面路径）
new_root = '/data/sunzhihao2/turuirui/dataset/arctic/data/data'
# 加载
with open(old_cache,'rb') as f:
    data = pickle.load(f)
# data 可为 dict 或 {'records': dict}
records = data.get('records', data) if isinstance(data, dict) else data
# 获取一个示例键，用于推断旧前缀
keys = list(records.keys())
if len(keys) == 0:
    print('原相机缓存没有记录。退出。'); sys.exit(1)
sample = keys[0]
# 找到 '/images' 在 sample 中的位置并截取旧前缀
idx = sample.find('/images')
if idx == -1:
    print('无法在样例键中找到 /images，手动指定替换前缀。样例键:', sample); sys.exit(1)
old_root = sample[:idx]
print('替换前缀:')
print(' old_root=', old_root)
print(' new_root=', new_root)
# 构建新 records
new_records = {}
for k,v in records.items():
    if k.startswith(old_root):
        new_k = new_root + k[idx:]
    else:
        # 如果不是以 old_root 开头，尝试保留原键
        new_k = k
    new_records[new_k] = v
# 保存（保持原始格式）
out = {'records': new_records} if 'records' in data else new_records
with open(new_cache,'wb') as f:
    pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
print('Saved remapped camera cache to', new_cache)
print('original count =', len(records), 'new count =', len(new_records))
