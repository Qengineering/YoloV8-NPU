#ifndef _RKNN_POSTPROCESS_H_
#define _RKNN_POSTPROCESS_H_

#include <stdint.h>
#include <vector>
#include "rknn_api.h"
#include "rk_common.h"

#define NMS_THRESH 0.45
#define BOX_THRESH 0.25
#define PROP_BOX_SIZE (5 + OBJ_CLASS_NUM)

int post_process(rknn_app_context_t *app_ctx, void *outputs, float conf_threshold, float nms_threshold, float scale_w, float scale_h, object_detect_result_list *od_results);

#endif //_RKNN_POSTPROCESS_H_
