import sys
path = sys.argv[1] if len(sys.argv) > 1 else 'MiVOLO/mivolo/model/mivolo_model.py'
content = open(path).read()
content = content.replace(
    '            drop_rate,\n            attn_drop_rate,',
    '            drop_rate,\n            0.0,\n            attn_drop_rate,'
)
open(path, 'w').write(content)
print('patched')