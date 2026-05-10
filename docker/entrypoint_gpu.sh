#!/bin/bash

# MiVOLO를 컨테이너 내부에 복사 후 패치 (호스트 파일 보호)
cp -r /app/MiVOLO /tmp/MiVOLO_gpu

sed -i 's/from timm.models._helpers import load_state_dict, remap_checkpoint/from timm.models._helpers import load_state_dict, remap_state_dict as remap_checkpoint/' /tmp/MiVOLO_gpu/mivolo/model/create_timm_model.py

sed -i '/from timm.models._pretrained import PretrainedCfg, split_model_name_tag/c\from timm.models._pretrained import PretrainedCfg\nfrom timm.models import split_model_name_tag' /tmp/MiVOLO_gpu/mivolo/model/create_timm_model.py

python /app/docker/patch_mivolo.py /tmp/MiVOLO_gpu/mivolo/model/mivolo_model.py

sed -i 's/block_dpr = drop_path_rate \*/block_dpr = float(drop_path_rate) */' \
    /usr/local/lib/python3.11/dist-packages/timm/models/volo.py

exec "$@"